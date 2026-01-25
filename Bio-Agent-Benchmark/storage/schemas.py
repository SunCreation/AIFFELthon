from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal
from datetime import datetime


@dataclass
class BenchmarkResult:
    task_id: str
    benchmark_name: str
    status: Literal["success", "failure", "error"]
    score: float = 0.0
    prediction: Optional[str] = None
    ground_truth: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
