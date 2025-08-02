#!/usr/bin/env python3
"""
Debug the specific tokens that might be causing the fragmentation
"""

def analyze_remaining_tokens():
    """Analyze where ##1, ##w, 72 might be coming from"""
    
    cleaned_text = "TRANSACTION_START CARD_INSERTED ATR_RECEIVED_T_0 OPCODE_FI CardNumber PIN_ENTERED OPCODE_IB CardNumber M-02, R-10011 DEVICE_ERROR ESC_000 VAL_000 REF_000 REJECTS_000 CARD_TAKEN TRANSACTION_END"
    
    print("=== TOKENIZATION FRAGMENTATION ANALYSIS ===")
    print()
    print("Cleaned text:")
    print(cleaned_text)
    print()
    
    # Split into tokens
    tokens = cleaned_text.split()
    
    print("Individual tokens:")
    for i, token in enumerate(tokens):
        print(f"{i+1:2d}. '{token}'")
    print()
    
    # Analyze potential fragmentation sources
    print("POTENTIAL FRAGMENTATION SOURCES:")
    print()
    
    # Check M-02,
    print("1. 'M-02,' token:")
    print(f"   - Contains hyphen: {'-' in 'M-02,'}")
    print(f"   - Contains comma: {',' in 'M-02,'}")
    print(f"   - BERT might split this into: ['M', '-', '##02', ','] or ['M', '##-', '##02', ',']")
    print()
    
    # Check R-10011
    print("2. 'R-10011' token:")
    print(f"   - Contains hyphen: {'-' in 'R-10011'}")
    print(f"   - Contains numbers: {'10011' in 'R-10011'}")
    print(f"   - BERT might split this into: ['R', '-', '##10011'] or ['R', '##-', '##1', '##0011']")
    print()
    
    # The real issue might be these weren't added to custom vocabulary
    print("ROOT CAUSE ANALYSIS:")
    print("✅ Source noise patterns (*7231*1*(Iw(1*3,) removed successfully")
    print("❌ But M-02, and R-10011 tokens are likely being fragmented by BERT's tokenizer")
    print()
    print("SOLUTIONS:")
    print("1. Add 'M-02' and 'R-10011' to custom vocabulary")
    print("2. OR: Clean these tokens further in preprocessing")
    print("3. OR: Replace with compound tokens like M_02 and R_10011")
    print()
    
    # Check what ##w could come from
    print("3. '##w' token analysis:")
    print("   - Could come from 'Iw' being split into ['I', '##w']")
    print("   - Our regex should have removed '*7231*1*(Iw(1*3,' entirely")
    print("   - This suggests either:")
    print("     a) The API isn't using updated preprocessing")
    print("     b) There's a different 'w' pattern we missed")
    print("     c) The heatmap is from old data")

if __name__ == "__main__":
    analyze_remaining_tokens()
