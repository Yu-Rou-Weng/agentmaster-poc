"""
A2A (Agent-to-Agent) Protocol Implementation
Based on Google's A2A specification - simplified for PoC
Ref: AgentMaster paper (Stanford / George Mason University)
"""

import uuid
import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Any


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"


@dataclass
class A2AMessage:
    """A2A protocol message between agents"""
    sender: str
    receiver: str
    role: MessageRole
    content: str
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass
class A2ATask:
    """A2A task unit dispatched from Orchestrator to domain agents"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_task_id: Optional[str] = None
    assigned_agent: str = ""
    description: str = ""
    state: TaskState = TaskState.SUBMITTED
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    messages: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    def add_message(self, msg: A2AMessage):
        self.messages.append(msg.to_dict())

    def complete(self, result: Any):
        self.state = TaskState.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str):
        self.state = TaskState.FAILED
        self.result = {"error": error}
        self.completed_at = time.time()


class A2ARouter:
    """Routes A2A messages between registered agents"""

    def __init__(self):
        self.agents = {}
        self.task_log = []

    def register_agent(self, name: str, agent_callable):
        self.agents[name] = agent_callable

    def dispatch(self, task: A2ATask) -> A2ATask:
        """Dispatch a task to the assigned agent via A2A protocol"""
        agent_name = task.assigned_agent
        if agent_name not in self.agents:
            task.fail(f"Agent '{agent_name}' not registered")
            return task

        # Log the A2A message exchange
        dispatch_msg = A2AMessage(
            sender="orchestrator",
            receiver=agent_name,
            role=MessageRole.USER,
            content=task.description,
            task_id=task.task_id,
        )
        task.add_message(dispatch_msg)
        task.state = TaskState.WORKING

        try:
            result = self.agents[agent_name](task.description)
            response_msg = A2AMessage(
                sender=agent_name,
                receiver="orchestrator",
                role=MessageRole.AGENT,
                content=str(result),
                task_id=task.task_id,
            )
            task.add_message(response_msg)
            task.complete(result)
        except Exception as e:
            task.fail(str(e))

        self.task_log.append(task.to_dict())
        return task
