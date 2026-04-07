import re


class PromptFilter:
    def __init__(self):
        # 🔴 Malicious / injection rules
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

            # 🔥 NEW INTENT-BASED RULES
            r"scam\s+people",
            r"how\s+to\s+(hack|scam|steal|fraud)",
            r"commit\s+(fraud|crime)",
            r"phishing\s+(attack|method)",
            r"social\s+engineering",
            r"exploit\s+(system|vulnerability)",

            # 🚫 Specifically block attempts to get others' passwords/credentials
            r"(get|obtain|steal|take)\s+(their|someone'?s|users?)\s+(password|credentials?)",
            r"(trick|convince|make|get)\s+(someone|users?|people)\s+(to\s+)?(give|share|reveal)",
            r"how\s+to\s+get\s+(someone|users?)\s+to\s+(give|share|reveal)\s+(their\s+)?(password|credentials?)",
        ]

        # 🔴 Sensitive data keywords
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

        # ✅ Safe prompts that should NOT be blocked
        self.safe_patterns = [
            r"create\s+(a\s+)?(secure|strong)?\s*password",
            r"generate\s+(a\s+)?password",
            r"password\s+(best\s+practices|tips|security)",
            r"how\s+to\s+(make|create|generate)\s+(a\s+)?password",
            r"how\s+to\s+prevent\s+phishing",
            r"phishing\s+awareness",
            r"security\s+best\s+practices"
        ]

    def check_prompt(self, prompt):
        text = prompt.lower()
        matched_rules = []
        safe_match = False

        # ✅ Check safe patterns first
        for pattern in self.safe_patterns:
            if re.search(pattern, text):
                safe_match = True
                break

        # 🔴 Only check blocking rules if NOT safe
        if not safe_match:
            for pattern in self.injection_patterns:
                if re.search(pattern, text):
                    matched_rules.append(f"injection rule matched: {pattern}")

            for pattern in self.sensitive_patterns:
                if re.search(pattern, text):
                    matched_rules.append(f"sensitive-data rule matched: {pattern}")

        # 🚫 Block only if matched rules exist AND not safe
        blocked = len(matched_rules) > 0
        if safe_match:
            blocked = False  # ✅ Safe prompts override blocking

        return {
            "prompt": prompt,
            "blocked": blocked,
            "reasons": matched_rules,
            "safe_override": safe_match
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
        elif result["safe_override"]:
            print("Safe prompt detected. No blocking applied.")
        else:
            print("No suspicious pattern detected.")

        print()