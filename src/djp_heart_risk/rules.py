"""Stub boundary for clinician-owned rules. Contains NO medical logic or thresholds."""
from dataclasses import dataclass

@dataclass(frozen=True)
class ClinicianRuleResult:
    emergency_flag: bool
    rule_set_version: str = "mock-clinician-fixture-v1"

class ClinicianRulesPort:
    """Production implementation must be supplied and approved by qualified clinicians."""
    def accept_mock_result(self, result: ClinicianRuleResult) -> ClinicianRuleResult:
        # Pass-through only: never derive, suppress, or override from ML probability.
        return result
