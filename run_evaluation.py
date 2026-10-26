import os
import re
import sys
import io
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))

from evaluate_sarkari_sathi import TEST_CASES
from assistant.ai_service import answer_question


def normalize_digits(text):
    """Convert Devanagari digits to English digits."""
    devanagari = "०१२३४५६७८९"
    english = "0123456789"
    return text.translate(str.maketrans(devanagari, english))


def normalize_amount(text):
    """Keep digits only, so comma formatting does not affect comparison."""
    return re.sub(r"\D", "", normalize_digits(text))


def mentions_missing_information(answer):
    """Recognize common English and Nepali uncertainty statements."""
    answer = answer.lower()

    phrases = [
        "does not identify",
        "does not mention",
        "there is no mention of",
        "does not state the exact name",
        "does not provide the name",
        "does not establish",
        "not specified",
        "does not specify",
        "cannot determine",
        "not mentioned",
        "not provided",
        "not available in the evidence",
        "information is absent",
        "evidence does not",
        "not established by the evidence",
        "नाम उल्लेख छैन",
        "उल्लेख गरेको छैन",
        "उपलब्ध छैन",
    ]
    return any(phrase in answer for phrase in phrases)


def check_report_findings(answer):
    """Check for concepts using alternative wording."""
    text = answer.lower()

    records_missing = any(term in text for term in [
        "records", "record", "not maintained",
        "not kept", "not available", "अभिलेख"
    ])

    implementation_issue = any(term in text for term in [
        "not implemented",
        "not even implemented",
        "not even been implemented",
        "not been implemented",
        "not found to have been implemented",
        "not carried out",
        "not put into practice",
        "कार्यान्वयन भएन",
        "कार्यान्वयन नभएको",
    ])

    return records_missing, implementation_issue


def run_test(test):
    print("\n" + "=" * 65)
    print(f"TEST {test['id']}: {test['question']}")
    print("=" * 65)

    try:
        result = answer_question(test["question"], language="en")
    except Exception as exc:
        print("ERROR:", type(exc).__name__, str(exc))
        return {
            "id": test["id"],
            "passed": False,
            "reason": "The answer function raised an error."
        }

    answer = result.get("answer", "")
    evidence = result.get("evidence", [])
    pages = sorted({
        item["page"] for item in evidence
        if isinstance(item, dict) and "page" in item
    })

    print("\nANSWER:\n", answer)
    print("\nRETRIEVED PAGES:", pages)

    checks = {}

    if test["id"] in (1, 2):
        expected_digits = normalize_amount(test["expected_answer"])
        answer_digits = normalize_amount(answer)

        # Check whether the expected numeric amount appears in the answer.
        checks["expected_amount_found"] = (
            bool(expected_digits) and expected_digits in answer_digits
        )

        checks["expected_page_retrieved"] = (
            bool(set(test["expected_pages"]) & set(pages))
        )

    elif test["id"] in (3, 5):
        checks["acknowledges_missing_information"] = (
            mentions_missing_information(answer)
        )

    elif test["id"] == 4:
        records_missing, implementation_issue = check_report_findings(answer)
        checks["mentions_missing_records"] = records_missing
        checks["mentions_implementation_issue"] = implementation_issue
        checks["expected_page_retrieved"] = (
            bool(set(test["expected_pages"]) & set(pages))
        )

    print("\nCHECKS:")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'REVIEW'}")

    # A test passes only when every automated check for it passes.
    passed = bool(checks) and all(checks.values())

    return {
        "id": test["id"],
        "passed": passed,
        "checks": checks,
        "pages": pages,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    results = []

    for test in TEST_CASES:
        results.append(run_test(test))

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    for result in results:
        status = "PASS" if result["passed"] else "REVIEW"
        print(f"Test {result['id']}: {status}")

    passed_count = sum(result["passed"] for result in results)
    print(f"\nAutomated checks passed: {passed_count}/{len(results)}")
    print("Review any failed checks against the actual answer.")
    print("This is a small test suite, not a formal accuracy measurement.")
