"""
High-quality RAG chunker for Mutual Fund FAQ chatbot.
Reads raw scheme pages and produces optimized chunks for RAG embeddings.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: strip and collapse multiple spaces.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Collapse multiple whitespace characters to single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_repetition(text: str) -> str:
    """
    Remove repetitive content and broken formatting.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text without repetition
    """
    if not text:
        return ""
    
    # Remove repeated consecutive sentences (same sentence appearing 2+ times)
    sentences = text.split('. ')
    if len(sentences) > 1:
        seen = set()
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Normalize for comparison (lowercase, remove extra spaces)
            sentence_key = re.sub(r'\s+', ' ', sentence.lower().strip())
            if sentence_key and sentence_key not in seen:
                seen.add(sentence_key)
                cleaned_sentences.append(sentence)
        
        # Only return cleaned if we actually removed something (to avoid over-aggressive cleaning)
        if len(cleaned_sentences) < len(sentences) * 0.8:  # More than 20% reduction
            text = '. '.join(cleaned_sentences)
    
    # Remove duplicate paragraphs (common in scraped content)
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        seen_paras = set()
        cleaned_paras = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_key = re.sub(r'\s+', ' ', para.lower().strip()[:100])  # First 100 chars for comparison
            if para_key and para_key not in seen_paras:
                seen_paras.add(para_key)
                cleaned_paras.append(para)
        if len(cleaned_paras) < len(paragraphs):
            text = '\n\n'.join(cleaned_paras)
    
    # Remove broken formatting patterns
    # Remove standalone punctuation or numbers
    text = re.sub(r'\s+[.!?]\s*[.!?]+', '. ', text)
    # Remove multiple consecutive punctuation
    text = re.sub(r'[.!?]{3,}', '.', text)
    # Fix broken spacing around punctuation
    text = re.sub(r'\s+([,.!?:;])', r'\1', text)
    text = re.sub(r'([,.!?:;])\s*([A-Z])', r'\1 \2', text)
    
    return normalize_whitespace(text)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for text (approximation for chunking).
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple approximation: character count / average chars per token
    return int(len(text) / chars_per_token)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving sentence boundaries.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Normalize whitespace first
    text = normalize_whitespace(text)
    
    # Split on sentence-ending punctuation followed by space or end of string
    # This regex handles: . ! ? followed by space or end of line
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\s*$)'
    sentences = re.split(sentence_pattern, text)
    
    # Filter out empty sentences and strip
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If no sentence boundaries found, return the whole text as one sentence
    if not sentences:
        return [text] if text else []
    
    return sentences


def chunk_text_by_tokens(
    text: str,
    min_tokens: int = 150,
    max_tokens: int = 300,
    chars_per_token: float = 4.0
) -> List[str]:
    """
    Split text into chunks of approximately 150-300 tokens.
    Keeps sentences intact.
    
    Args:
        text: Input text to chunk
        min_tokens: Minimum tokens per chunk (target: 150)
        max_tokens: Maximum tokens per chunk (target: 300)
        chars_per_token: Approximate characters per token (default 4.0)
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean repetition and normalize
    text = clean_repetition(text)
    text = normalize_whitespace(text)
    
    # Calculate character limits based on token targets
    min_chars = int(min_tokens * chars_per_token)  # ~600 chars for 150 tokens
    max_chars = int(max_tokens * chars_per_token)  # ~1200 chars for 300 tokens
    
    # If text is shorter than max, return as single chunk if it meets minimum
    estimated_tokens = estimate_tokens(text, chars_per_token)
    if estimated_tokens <= max_tokens:
        if estimated_tokens >= min_tokens or estimated_tokens > 0:
            return [text]
        return []  # Too short, skip
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence, chars_per_token)
        
        # If single sentence exceeds max_tokens, split it further by words (fallback)
        if sentence_tokens > max_tokens:
            # Save current chunk if it exists
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                estimated_chunk_tokens = estimate_tokens(chunk_text, chars_per_token)
                if estimated_chunk_tokens >= min_tokens or not chunks:
                    chunks.append(chunk_text)
                current_chunk = []
                current_token_count = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_chunk_tokens = 0
            
            for word in words:
                word_tokens = estimate_tokens(word + ' ', chars_per_token)
                
                if word_chunk_tokens + word_tokens > max_tokens and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append(chunk_text)
                    word_chunk = [word]
                    word_chunk_tokens = estimate_tokens(word, chars_per_token)
                else:
                    word_chunk.append(word)
                    word_chunk_tokens += word_tokens
            
            if word_chunk:
                # Add remaining words as new current chunk
                current_chunk = word_chunk
                current_token_count = word_chunk_tokens
        
        # If adding this sentence would exceed max_tokens, start new chunk
        elif current_token_count + sentence_tokens > max_tokens:
            if current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                estimated_chunk_tokens = estimate_tokens(chunk_text, chars_per_token)
                
                if estimated_chunk_tokens >= min_tokens or not chunks:
                    chunks.append(chunk_text)
                else:
                    # Try to merge with previous chunk if too small
                    if chunks:
                        prev_chunk_tokens = estimate_tokens(chunks[-1], chars_per_token)
                        combined_tokens = prev_chunk_tokens + estimated_chunk_tokens
                        if combined_tokens <= max_tokens * 1.5:  # Allow slight overflow
                            chunks[-1] = chunks[-1] + ' ' + chunk_text
                        else:
                            chunks.append(chunk_text)
                
                # Start new chunk with current sentence
                current_chunk = [sentence]
                current_token_count = sentence_tokens
            else:
                # Single sentence chunk (if it meets minimum)
                if sentence_tokens >= min_tokens:
                    chunks.append(sentence)
                current_chunk = []
                current_token_count = 0
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        estimated_chunk_tokens = estimate_tokens(chunk_text, chars_per_token)
        
        if estimated_chunk_tokens >= min_tokens or not chunks:
            chunks.append(chunk_text)
        elif chunks:
            # Try to merge with previous chunk if too small
            prev_chunk_tokens = estimate_tokens(chunks[-1], chars_per_token)
            combined_tokens = prev_chunk_tokens + estimated_chunk_tokens
            if combined_tokens <= max_tokens * 1.5:
                chunks[-1] = chunks[-1] + ' ' + chunk_text
            else:
                chunks.append(chunk_text)
    
    # Filter out empty chunks and clean each chunk
    final_chunks = []
    for chunk in chunks:
        chunk = clean_repetition(chunk)
        chunk = normalize_whitespace(chunk)
        if chunk and chunk.strip():
            final_chunks.append(chunk)
    
    return final_chunks


def remove_section_text_from_clean_text(
    text_clean: str,
    sections: Dict[str, Optional[str]]
) -> str:
    """
    Remove section text from text_clean to avoid duplication.
    Uses conservative matching to find and remove section content.
    
    Args:
        text_clean: Clean text from the page
        sections: Dictionary of section keys and values
        
    Returns:
        Text with section content removed
    """
    if not text_clean:
        return ""
    
    result = text_clean
    text_lower = text_clean.lower()
    
    # For each non-empty section, try to remove its content from text_clean
    for section_key, section_value in sections.items():
        if section_value and section_value.strip():
            section_text = normalize_whitespace(section_value)
            section_text_lower = section_text.lower()
            
            # Only remove if section text is substantial (at least 15 chars)
            # and appears as a contiguous substring (case-insensitive)
            if len(section_text) >= 15:
                # Find first occurrence (case-insensitive)
                idx = text_lower.find(section_text_lower)
                if idx != -1:
                    # Remove from original text (preserving case)
                    # Remove the matched length from original text
                    result = result[:idx] + result[idx + len(section_text):]
                    # Update lowercase version for next iteration
                    text_lower = result.lower()
    
    # Normalize whitespace after removal
    result = normalize_whitespace(result)
    
    return result


def generate_chunk_id(
    scheme_name: str,
    section: Optional[str],
    chunk_index: int
) -> str:
    """
    Generate a unique chunk ID.
    
    Args:
        scheme_name: Name of the scheme
        section: Section name (or None for text_clean chunks)
        chunk_index: Index of the chunk
        
    Returns:
        Unique chunk ID
    """
    # Normalize scheme name for ID
    scheme_normalized = re.sub(r'[^a-z0-9]+', '_', scheme_name.lower())
    scheme_normalized = re.sub(r'_+', '_', scheme_normalized).strip('_')
    
    if section:
        section_normalized = section.lower().replace(' ', '_')
        chunk_id = f"{scheme_normalized}_{section_normalized}_{chunk_index}"
    else:
        chunk_id = f"{scheme_normalized}_text_{chunk_index}"
    
    return chunk_id


def load_raw_records(path: str) -> List[Dict]:
    """
    Load raw records from JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of record dictionaries
    """
    records = []
    file_path = Path(path)
    
    if not file_path.exists():
        print(f"Warning: File {path} does not exist.")
        return records
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {path}: {e}")
                    continue
        
        print(f"Loaded {len(records)} records from {path}")
        
    except Exception as e:
        print(f"Error loading records from {path}: {e}")
        raise
    
    return records


def generate_chunks_for_record(record: Dict) -> List[Dict]:
    """
    Generate chunks for a single record from structured fields.
    
    Args:
        record: Single record dictionary from raw_scheme_pages.jsonl
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    # Extract record fields
    scheme_name = record.get('scheme_name', 'Unknown Scheme')
    amc_name = record.get('amc_name', '')
    source_url = record.get('source_url', '')
    last_scraped_at = record.get('last_scraped_at', '')
    
    # Build comprehensive text chunks from structured data
    chunk_texts = []
    
    # Chunk 1: Basic Information
    basic_info_parts = [f"Scheme: {scheme_name}"]
    if amc_name:
        basic_info_parts.append(f"AMC: {amc_name}")
    if record.get('category'):
        basic_info_parts.append(f"Category: {record['category']}")
    if record.get('plan_type'):
        basic_info_parts.append(f"Plan Type: {record['plan_type']}")
    if record.get('option'):
        basic_info_parts.append(f"Option: {record['option']}")
    if record.get('risk_level'):
        basic_info_parts.append(f"Risk Level: {record['risk_level']}")
    
    if len(basic_info_parts) > 1:
        chunk_texts.append('. '.join(basic_info_parts) + '.')
    
    # Chunk 2: Fund Size and NAV
    fund_info_parts = []
    if record.get('fund_size_cr'):
        fund_info_parts.append(f"Fund Size (AUM): ₹{record['fund_size_cr']} Crore")
    if record.get('nav_value'):
        fund_info_parts.append(f"NAV: ₹{record['nav_value']}")
    if record.get('nav_date'):
        fund_info_parts.append(f"NAV Date: {record['nav_date']}")
    
    if fund_info_parts:
        chunk_texts.append(f"{scheme_name} - " + '. '.join(fund_info_parts) + '.')
    
    # Chunk 3: Investment Minimums
    investment_parts = []
    if record.get('min_sip_amount'):
        investment_parts.append(f"Minimum SIP: ₹{record['min_sip_amount']}")
    if record.get('min_lumpsum_first_investment'):
        investment_parts.append(f"Minimum Lumpsum Investment: ₹{record['min_lumpsum_first_investment']}")
    if record.get('min_lumpsum_additional'):
        investment_parts.append(f"Minimum Additional Investment: ₹{record['min_lumpsum_additional']}")
    
    if investment_parts:
        chunk_texts.append(f"{scheme_name} - " + '. '.join(investment_parts) + '.')
    
    # Chunk 4: Returns
    returns_parts = []
    if record.get('returns_1y_percent') is not None:
        returns_parts.append(f"1 Year Return: {record['returns_1y_percent']}%")
    if record.get('returns_3y_percent') is not None:
        returns_parts.append(f"3 Year Return: {record['returns_3y_percent']}%")
    if record.get('returns_5y_percent') is not None:
        returns_parts.append(f"5 Year Return: {record['returns_5y_percent']}%")
    if record.get('returns_all_percent') is not None:
        returns_parts.append(f"Returns Since Inception: {record['returns_all_percent']}%")
    
    if returns_parts:
        chunk_texts.append(f"{scheme_name} - " + '. '.join(returns_parts) + '.')
    
    # Chunk 5: Fees and Charges
    fees_parts = []
    if record.get('expense_ratio_percent') is not None:
        fees_parts.append(f"Expense Ratio (TER): {record['expense_ratio_percent']}%")
    if record.get('exit_load_text'):
        exit_load = record['exit_load_text'][:200]  # Limit length
        fees_parts.append(f"Exit Load: {exit_load}")
    if record.get('stamp_duty_text'):
        fees_parts.append(f"Stamp Duty: {record['stamp_duty_text'][:100]}")
    
    if fees_parts:
        chunk_texts.append(f"{scheme_name} - " + '. '.join(fees_parts) + '.')
    
    # Chunk 6: Fund Manager
    if record.get('fund_manager_name'):
        fm_text = f"{scheme_name} - Fund Manager: {record['fund_manager_name']}"
        if record.get('fund_manager_since'):
            fm_text += f". Managing since: {record['fund_manager_since']}"
        chunk_texts.append(fm_text + '.')
    
    # Chunk 7: Ratios and Metrics
    ratios_parts = []
    if record.get('pe_ratio') is not None:
        ratios_parts.append(f"P/E Ratio: {record['pe_ratio']}")
    if record.get('pb_ratio') is not None:
        ratios_parts.append(f"P/B Ratio: {record['pb_ratio']}")
    if record.get('alpha') is not None:
        ratios_parts.append(f"Alpha: {record['alpha']}")
    if record.get('beta') is not None:
        ratios_parts.append(f"Beta: {record['beta']}")
    if record.get('sharpe_ratio') is not None:
        ratios_parts.append(f"Sharpe Ratio: {record['sharpe_ratio']}")
    
    if ratios_parts:
        chunk_texts.append(f"{scheme_name} - " + '. '.join(ratios_parts) + '.')
    
    # Create chunk dictionaries
    chunk_index = 1
    for text in chunk_texts:
        if text and len(text) >= 20:  # Skip very short chunks
            chunk_id = generate_chunk_id(scheme_name, None, chunk_index)
            chunk_index += 1
            
            chunk = {
                'chunk_id': chunk_id,
                'scheme_name': scheme_name,
                'section': 'structured_data',
                'text': text,
                'source_url': source_url,
                'last_scraped_at': last_scraped_at
            }
            chunks.append(chunk)
    
    return chunks


def save_chunks(chunks: List[Dict], output_path: str) -> None:
    """
    Save chunks to JSONL file.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output JSONL file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json_line = json.dumps(chunk, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
        
    except Exception as e:
        print(f"Error saving chunks to {output_path}: {e}")
        raise


def main() -> None:
    """Main function to process raw records and generate chunks."""
    input_file = "data/raw_scheme_pages.jsonl"
    output_file = "data/processed_chunks.jsonl"
    
    print("=" * 60)
    print("CHUNKER - Processing Raw Scheme Data")
    print("=" * 60)
    
    # Load raw records
    print(f"\nLoading records from {input_file}...")
    records = load_raw_records(input_file)
    
    if not records:
        print("No records found. Please run scraper.py first.")
        return
    
    # Generate chunks for all records
    print(f"\nGenerating chunks from {len(records)} records...")
    all_chunks = []
    
    for i, record in enumerate(records, 1):
        scheme_name = record.get('scheme_name', 'Unknown')
        print(f"  Processing [{i}/{len(records)}]: {scheme_name}")
        
        try:
            chunks = generate_chunks_for_record(record)
            all_chunks.extend(chunks)
            print(f"    Generated {len(chunks)} chunks")
        except Exception as e:
            print(f"    Error processing {scheme_name}: {e}")
            continue
    
    # Save chunks
    print(f"\nSaving chunks to {output_file}...")
    save_chunks(all_chunks, output_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Schemes processed:     {len(records)}")
    print(f"# Chunks created:      {len(all_chunks)}")
    print(f"Avg chunks/scheme:     {len(all_chunks) / len(records):.1f}" if records else "N/A")
    print(f"Output file:           {output_file}")
    print("=" * 60)
    
    # Show chunk statistics
    if all_chunks:
        chunk_sizes = [len(chunk.get('text', '')) for chunk in all_chunks]
        estimated_tokens = [estimate_tokens(chunk.get('text', '')) for chunk in all_chunks]
        
        print(f"\nChunk statistics:")
        print(f"  Total chunks:        {len(all_chunks)}")
        print(f"  Chunk size (chars):  Min={min(chunk_sizes)}, Max={max(chunk_sizes)}, Avg={sum(chunk_sizes) / len(chunk_sizes):.0f}")
        print(f"  Estimated tokens:    Min={min(estimated_tokens)}, Max={max(estimated_tokens)}, Avg={sum(estimated_tokens) / len(estimated_tokens):.0f}")
        
        # Count chunks by type
        section_chunks = sum(1 for chunk in all_chunks if chunk.get('section') is not None)
        text_chunks = len(all_chunks) - section_chunks
        print(f"  Section chunks:      {section_chunks}")
        print(f"  Text chunks:         {text_chunks}")


if __name__ == "__main__":
    main()

