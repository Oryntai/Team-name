"""Development wiring only: absence of ML must always fail explicitly."""

from backend.contracts import AnalysisUnavailable


class UnavailableAnalyzer:
    def analyze(self, context, video):
        raise AnalysisUnavailable("No local ML analyzer configured")


class UnavailableRules:
    def evaluate(self, observations, context):
        raise AnalysisUnavailable("No local rule engine configured")


def create_analyzer():
    return UnavailableAnalyzer()


def create_rules():
    return UnavailableRules()
