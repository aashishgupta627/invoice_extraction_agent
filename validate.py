import re
from typing import Dict, List, Any

# State code mapping (first two digits of GSTIN)
STATE_CODES = {
    "01": "J&K", "02": "HP", "03": "PB", "04": "CH", "05": "UK", "06": "HR",
    "07": "DL", "08": "RJ", "09": "UP", "10": "BI", "11": "SK", "12": "AR",
    "13": "NL", "14": "MN", "15": "MZ", "16": "TR", "17": "ML", "18": "AS",
    "19": "WB", "20": "JH", "21": "OD", "22": "CG", "23": "MP", "24": "GJ",
    "25": "MH", "26": "KA", "27": "GA", "28": "TN", "29": "AP", "30": "TL",
    "31": "LA", "32": "KL", "33": "PY", "34": "AN", "35": "DN", "36": "DD",
    "37": "LD"
}

def check_gstin_format(gstin: str) -> bool:
    """Rule 1: GSTIN format."""
    if not gstin:
        return False
    pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$'
    return bool(re.match(pattern, gstin))

def derive_state_from_gstin(gstin: str) -> str:
    """Extract state code from first two digits."""
    if len(gstin) >= 2:
        return STATE_CODES.get(gstin[:2], "")
    return ""

def check_state_tax_rule(seller_gstin: str, buyer_gstin: str,
                         tax_breakup: Dict[str, float]) -> List[str]:
    """Rule 2 & 3: State derivation and tax type."""
    flags = []
    if not seller_gstin or not buyer_gstin:
        flags.append("Missing GSTIN for tax rule check")
        return flags
    
    seller_state = derive_state_from_gstin(seller_gstin)
    buyer_state = derive_state_from_gstin(buyer_gstin)
    if not seller_state or not buyer_state:
        flags.append("Invalid state code from GSTIN")
        return flags
    
    cgst = tax_breakup.get("cgst", 0.0)
    sgst = tax_breakup.get("sgst", 0.0)
    igst = tax_breakup.get("igst", 0.0)
    
    if seller_state == buyer_state:
        # Expect CGST+SGST, IGST≈0
        if abs(igst) > 0.01:
            flags.append(f"IGST should be zero for intra-state, got {igst}")
        if cgst <= 0 or sgst <= 0:
            flags.append("CGST/SGST missing for intra-state")
    else:
        # Expect IGST, CGST/SGST≈0
        if abs(cgst) > 0.01 or abs(sgst) > 0.01:
            flags.append("CGST/SGST should be zero for inter-state")
        if igst <= 0:
            flags.append("IGST missing for inter-state")
    return flags

def check_arithmetic(line_items: List[Dict], tax_breakup: Dict[str, float],
                     total_amount: float) -> List[str]:
    """Rule 4: Arithmetic consistency."""
    flags = []
    taxable_sum = sum(item.get("amount", 0.0) for item in line_items)
    taxable_val = tax_breakup.get("taxable_value", 0.0)
    if abs(taxable_sum - taxable_val) > 1.0:
        flags.append(f"Line items sum ({taxable_sum}) != taxable_value ({taxable_val})")
    
    total_gst = tax_breakup.get("total_gst", 0.0)
    computed_total = taxable_val + total_gst
    if abs(computed_total - total_amount) > 1.0:
        flags.append(f"taxable+total_gst ({computed_total}) != total_amount ({total_amount})")
    return flags

def check_missing_fields(record: Dict) -> List[str]:
    """Rule 5: Required fields."""
    flags = []
    required = [
        ("seller", "gstin"), ("buyer", "gstin"),
        ("invoice_number",), ("invoice_date",), ("total_amount",)
    ]
    for path in required:
        value = record
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else None
        if not value:
            flags.append(f"Missing required field: {'.'.join(path)}")
    return flags

def validate_invoice(record: Dict) -> Dict:
    """Run all five rules and return updated record with validation_flags."""
    flags = []
    
    # Rule 1: GSTIN format
    for party in ["seller", "buyer"]:
        gstin = record.get(party, {}).get("gstin", "")
        if not check_gstin_format(gstin):
            flags.append(f"{party} GSTIN format invalid: {gstin}")
    
    # Rule 2 & 3: State code and tax type
    seller_gstin = record.get("seller", {}).get("gstin", "")
    buyer_gstin = record.get("buyer", {}).get("gstin", "")
    tax_breakup = record.get("tax_breakup", {})
    flags.extend(check_state_tax_rule(seller_gstin, buyer_gstin, tax_breakup))
    
    # Rule 4: Arithmetic
    line_items = record.get("line_items", [])
    total_amount = record.get("total_amount", 0.0)
    flags.extend(check_arithmetic(line_items, tax_breakup, total_amount))
    
    # Rule 5: Missing fields
    flags.extend(check_missing_fields(record))
    
    # Deduplicate flags
    record["validation_flags"] = list(set(flags))
    return record