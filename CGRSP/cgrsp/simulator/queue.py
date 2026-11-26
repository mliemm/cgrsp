"""
Job Queue System Module

Implements the job queue with priority management based on deadlines
and job tracking for the CGRSP simulator.
"""

import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from enum import Enum


class JobState(str, Enum):
    """Job states in the system."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEADLINE_MISSED = "deadline_missed"


@dataclass
class Job:
    """
    Represents a rendering job in the system.

    Attributes:
        job_id: Unique job identifier
        complexity: Job complexity κ_j (affects execution time)
        deadline: Job deadline d_j (absolute time in seconds)
        submit_time: Job submission time t_j^submit
        estimated_duration: Estimated execution time (in seconds)
        state: Current job state
        assigned_node: Node ID if running, None otherwise
        queue_entry_time: Time when job entered the queue
        start_time: Time when job started execution
        completion_time: Time when job completed
        required_memory: VRAM requirement in GB (optional)
        preferred_resource: Preferred GPU type (optional)
    """

    job_id: int
    complexity: float  # κ_j
    deadline: float  # d_j (absolute time)
    submit_time: float  # t_j^submit
    estimated_duration: float  # seconds
    state: JobState = JobState.QUEUED
    assigned_node: Optional[int] = None
    queue_entry_time: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    required_memory: Optional[float] = None
    preferred_resource: Optional[str] = None
    # Realised complexity when it differs from the label schedulers see. None (the
    # canonical setting) means the label is exact; only the kappa-label sensitivity
    # study sets it.
    true_complexity: Optional[float] = None

    def __post_init__(self):
        """Initialize queue entry time if not set."""
        if self.queue_entry_time == 0.0:
            self.queue_entry_time = self.submit_time

    def get_waiting_time(self, current_time: float) -> float:
        """
        Calculate current waiting time.

        Args:
            current_time: Current simulation time

        Returns:
            Waiting time in seconds
        """
        if self.start_time is not None:
            return self.start_time - self.queue_entry_time
        else:
            return current_time - self.queue_entry_time

    def get_response_time(self, current_time: float) -> float:
        """
        Calculate current response time.

        Args:
            current_time: Current simulation time

        Returns:
            Response time in seconds (waiting + execution so far)
        """
        if self.completion_time is not None:
            return self.completion_time - self.submit_time
        elif self.start_time is not None:
            return current_time - self.submit_time
        else:
            return current_time - self.submit_time

    def get_remaining_time(self, current_time: float) -> float:
        """
        Calculate remaining execution time.

        Args:
            current_time: Current simulation time

        Returns:
            Remaining time in seconds
        """
        if self.state == JobState.COMPLETED:
            return 0.0
        elif self.state == JobState.RUNNING and self.start_time is not None:
            elapsed = current_time - self.start_time
            remaining = self.estimated_duration - elapsed
            return max(0.0, remaining)
        else:
            return self.estimated_duration

    def get_time_to_deadline(self, current_time: float) -> float:
        """
        Calculate time remaining until deadline.

        Args:
            current_time: Current simulation time

        Returns:
            Time to deadline in seconds (negative if past deadline)
        """
        return self.deadline - current_time

    def is_deadline_missed(self, current_time: float) -> bool:
        """
        Check if job has missed its deadline.

        Args:
            current_time: Current simulation time

        Returns:
            True if deadline is missed
        """
        return current_time > self.deadline

    def get_urgency(self, current_time: float) -> float:
        """
        Calculate job urgency score (time to deadline).

        Args:
            current_time: Current simulation time

        Returns:
            Urgency score (lower = more urgent)
        """
        return self.get_time_to_deadline(current_time)

    def __lt__(self, other: "Job") -> bool:
        """
        Compare jobs for priority queue (deadline-based priority).

        Jobs with earlier deadlines have higher priority (lower value).
        """
        return self.deadline < other.deadline


class JobQueue:
    """
    Priority queue for managing jobs waiting to be scheduled.

    Jobs are prioritized by deadline (earliest deadline first).
    """

    def __init__(self):
        """Initialize an empty job queue."""
        self._queue: List[Job] = []  # heapq-managed list
        self._job_map: dict[int, Job] = {}  # job_id -> Job mapping
        self._counter = 0  # For tie-breaking in priority
        self._removed_ids: set[int] = set()  # Track removed jobs

    def push(self, job: Job) -> None:
        """
        Add a job to the queue.

        Args:
            job: Job to add
        """
        # Check if job ID already exists
        if job.job_id in self._job_map:
            raise ValueError(f"Job ID {job.job_id} already exists in queue")

        # Add to heap with tie-breaker counter
        entry = (job.deadline, self._counter, job)
        heapq.heappush(self._queue, entry)
        self._job_map[job.job_id] = job
        self._counter += 1

    def pop(self) -> Optional[Job]:
        """
        Remove and return the highest priority job.

        Returns:
            Job with earliest deadline, or None if queue is empty
        """
        while self._queue:
            deadline, counter, job = heapq.heappop(self._queue)

            # Skip if job was marked as removed
            if job.job_id in self._removed_ids:
                self._removed_ids.remove(job.job_id)
                del self._job_map[job.job_id]
                continue

            # Valid job found - remove from job_map and return
            del self._job_map[job.job_id]
            return job

        return None

    def peek(self) -> Optional[Job]:
        """
        Return the highest priority job without removing it.

        Returns:
            Job with earliest deadline, or None if queue is empty
        """
        while self._queue:
            deadline, counter, job = self._queue[0]

            # Skip if job was marked as removed
            if job.job_id in self._removed_ids:
                heapq.heappop(self._queue)
                self._removed_ids.remove(job.job_id)
                del self._job_map[job.job_id]
                continue

            return job

        return None

    def remove(self, job_id: int) -> bool:
        """
        Remove a job from the queue by ID.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was found and marked for removal
        """
        if job_id not in self._job_map:
            return False

        # Mark for lazy removal
        self._removed_ids.add(job_id)
        return True

    def get_job(self, job_id: int) -> Optional[Job]:
        """
        Get a job by ID without removing it.

        Args:
            job_id: ID of job to retrieve

        Returns:
            Job if found, None otherwise
        """
        return self._job_map.get(job_id)

    def update_priority(self, job_id: int, new_deadline: float) -> bool:
        """
        Update a job's deadline (priority).

        Args:
            job_id: ID of job to update
            new_deadline: New deadline value

        Returns:
            True if job was found and updated
        """
        if job_id not in self._job_map or job_id in self._removed_ids:
            return False

        # The lazy-removal set is keyed by job_id only, so it cannot
        # distinguish the stale entry from a re-pushed one; physically drop the
        # old entry instead (O(n), acceptable for this rarely-used operation).
        self._queue = [e for e in self._queue if e[2].job_id != job_id]
        heapq.heapify(self._queue)

        # Get job and update deadline
        job = self._job_map[job_id]
        job.deadline = new_deadline

        # Add new entry with updated priority
        entry = (new_deadline, self._counter, job)
        heapq.heappush(self._queue, entry)
        self._counter += 1

        return True

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        # Account for jobs marked for removal
        return len(self) == 0

    def __len__(self) -> int:
        """Get number of jobs in queue."""
        # Exclude jobs marked for removal
        return len(self._job_map) - len(self._removed_ids)

    def get_queue_length(self) -> int:
        """Get current number of jobs in queue (alias for __len__)."""
        return len(self)

    def get_all_jobs(self) -> List[Job]:
        """
        Get all jobs in the queue (unsorted).

        Returns:
            List of all jobs
        """
        self._compact_removed()
        # Filter out removed jobs
        return [
            job for job_id, job in self._job_map.items()
            if job_id not in self._removed_ids
        ]

    def get_jobs_sorted_by_priority(self) -> List[Job]:
        """
        Get all jobs sorted by priority (deadline).

        Returns:
            List of jobs sorted by deadline (earliest first)
        """
        self._compact_removed()
        # Create a copy of the queue and pop all elements
        temp_queue = self._queue.copy()
        heapq.heapify(temp_queue)

        jobs = []
        while temp_queue:
            deadline, counter, job = heapq.heappop(temp_queue)
            if job.job_id not in self._removed_ids:
                jobs.append(job)

        return jobs

    def _compact_removed(self) -> None:
        """Bound lazy-removal storage by the live queue and a small batch buffer.

        Continuous calendars reuse one queue for the whole month, so retaining
        every dispatched job would make each later scheduling event scan history.
        Heap counters are retained to preserve equal-deadline ordering.
        """
        if len(self._removed_ids) <= max(64, len(self)):
            return
        self._queue = [entry for entry in self._queue if entry[2].job_id not in self._removed_ids]
        heapq.heapify(self._queue)
        for job_id in self._removed_ids:
            del self._job_map[job_id]
        self._removed_ids.clear()

    def get_jobs_deadline_imminent(
        self,
        current_time: float,
        threshold: float = 3600.0
    ) -> List[Job]:
        """
        Get jobs with deadlines within threshold seconds.

        Args:
            current_time: Current simulation time
            threshold: Time threshold in seconds (default: 1 hour)

        Returns:
            List of urgent jobs
        """
        return [
            job for job in self.get_all_jobs()
            if job.get_time_to_deadline(current_time) <= threshold
        ]

    def get_jobs_missed_deadline(self, current_time: float) -> List[Job]:
        """
        Get jobs that have already missed their deadlines.

        Args:
            current_time: Current simulation time

        Returns:
            List of jobs with missed deadlines
        """
        return [
            job for job in self.get_all_jobs()
            if job.is_deadline_missed(current_time)
        ]

    def clear(self) -> None:
        """Remove all jobs from the queue."""
        self._queue.clear()
        self._job_map.clear()
        self._removed_ids.clear()
        self._counter = 0
