import re


class PromptFilter:
    def __init__(self):
        self.injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"disregard\s+all\s+prior\s+rules",
            r"reveal\s+(confidential|private|secret)\s+(data|information|files)?",
            r"show\s+(me\s+)?(internal|hidden|private)\s+(instructions|data|documents)",
            r"bypass\s+(security|safeguards|restrictions)",
            r"leak\s+(sensitive|confidential|private)\s+data",
            r"print\s+(the\s+)?system\s+prompt",
            r"tell\s+me\s+the\s+hidden\s+prompt",
            r"extract\s+(credentials|passwords|tokens|keys)",
            r"give\s+me\s+(employee|customer|client)\s+(records|data|files)",

    # 🔥 NEW RULES (intent-based)
            r"scam\s+people",
            r"how\s+to\s+(hack|scam|steal|fraud)",
            r"commit\s+(fraud|crime)",
            r"phishing\s+(attack|method)",
            r"social\s+engineering",
            r"exploit\s+(system|vulnerability)"
                                ]
        self.sensitive_patterns = [
            r"passwords?",
            r"ssn|social\s+security\s+number",
            r"credit\s+card\s+number",
            r"bank\s+account",
            r"api\s+key",
            r"access\s+token",
            r"secret\s+key",
            r"private\s+key",
            r"confidential\s+information",
            r"personal\s+data",
            r"medical\s+records?",
            r"customer\s+records?",
            r"employee\s+records?"
        ]

    def check_prompt(self, prompt):
        text = prompt.lower()

        matched_rules = []

        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                matched_rules.append(f"injection rule matched: {pattern}")

        for pattern in self.sensitive_patterns:
            if re.search(pattern, text):
                matched_rules.append(f"sensitive-data rule matched: {pattern}")

        blocked = len(matched_rules) > 0

        return {
            "prompt": prompt,
            "blocked": blocked,
            "reasons": matched_rules
        }


if __name__ == "__main__":
    filter_system = PromptFilter()

    while True:
        user_prompt = input("Enter a prompt (or type quit): ")

        if user_prompt.lower() == "quit":
            break

        result = filter_system.check_prompt(user_prompt)

        print("\nResult:")
        print("Blocked:", result["blocked"])

        if result["reasons"]:
            print("Reasons:")
            for reason in result["reasons"]:
                print("-", reason)
        else:
            print("No suspicious pattern detected.")

        print()