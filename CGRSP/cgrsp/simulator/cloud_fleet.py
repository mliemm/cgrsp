"""
RunPod Cloud Fleet Manager

Simulates GPU availability on RunPod Secure Cloud without knowing the actual stock count.
Uses probabilistic models to determine wait times for on-demand GPU provisioning.
"""

import random
from typing import List, Dict, Optional, Any
from enum import Enum

from cgrsp.config import ResourceType, RESOURCE_SPECS, AvailabilityParameters
from cgrsp.simulator.fleet import FleetManager, Node, NodeState, UnavailableReason
from cgrsp.data.runpod_availability import RunPodAvailabilityModel, StockStatus

class CloudNodeState(str, Enum):
    """Extended states for cloud nodes."""
    PROVISIONING = "provisioning"  # Waiting for cloud allocation
    ACTIVE = "active"              # Rented and ready for jobs
    RUNNING = "running"            # Executing a job
    TERMINATED = "terminated"      # Released back to cloud

class RunPodCloudFleetManager(FleetManager):
    """
    Fleet manager for RunPod Cloud environments.
    
    Instead of hosting GPUs, this manager "rents" them from RunPod
    based on availability probabilities. It does not know the actual 
    number of GPUs in RunPod's stock.
    
    Attributes:
        max_concurrency: Maximum number of GPUs allowed for this user (quota)
        availability_model: Model for RunPod GPU availability
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        supported_types: Optional[List[ResourceType]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the cloud fleet manager.

        Args:
            max_concurrency: Max simultaneous GPUs allowed (user quota)
            supported_types: List of GPU types the user is willing to rent
            random_seed: Random seed for reproducibility
        """
        # Initialize with empty config but valid availability_params for base class compatibility
        super().__init__(
            fleet_config=[], 
            availability_params=AvailabilityParameters(),
            random_seed=random_seed
        )
        
        self.max_concurrency = max_concurrency
        self.supported_types = supported_types or ResourceType.supported_gpus()
        
        self.availability_model = RunPodAvailabilityModel(seed=random_seed)
        
        # Current qualitative stock status for each resource type
        self._stock_statuses: Dict[ResourceType, StockStatus] = {}
        
        # In cloud mode, we always have 'max_concurrency' potential slots
        # Distribute slots across supported types for a heterogeneous cloud fleet
        for i in range(self.max_concurrency):
            resource_type = self.supported_types[i % len(self.supported_types)]
            self._add_cloud_slot(resource_type)

    def update_stock_statuses(self, current_time: float):
        """Update qualitative stock status for all supported types.

        The status is drawn against the resource type's own beta_g, indexed on
        ``rt.value`` directly. Routing the type through a mapping to another GPU
        first would substitute the mapped GPU's beta_g (the A4500 would
        draw against the A5000's 0.65, the A4000 against the RTX 4090's 0.30),
        so the status a scheduler observes would come from a different
        availability model than the provisioning delay it subsequently incurs.
        The two are meant to be independent draws from the same model.
        """
        for rt in self.supported_types:
            try:
                self._stock_statuses[rt] = self.availability_model.get_stock_status(
                    rt.value, current_time
                )
            except (ValueError, KeyError):
                self._stock_statuses[rt] = StockStatus.HIGH

    def get_stock_status(self, resource_type: ResourceType) -> StockStatus:
        """Get the current qualitative stock status (High/Medium/Low) for a resource type."""
        return self._stock_statuses.get(resource_type, StockStatus.HIGH)

    def calculate_time_to_next_transition(self, node_id: int, current_time: float) -> float:
        """
        Cloud nodes don't follow the local Markov transition model.
        Returns infinity to disable simulator-driven transitions.
        """
        return float('inf')

    def execute_stochastic_transition(self, node_id: int, current_time: float) -> bool:
        """Disabled for cloud nodes."""
        return False

    def _add_cloud_slot(self, resource_type: ResourceType) -> Node:
        """Add a potential cloud instance slot."""
        node_id = self._next_node_id
        self._next_node_id += 1
        
        # Cloud nodes start as IDLE in the simulator's view so they can be assigned jobs,
        # but the actual assignment will trigger a "provisioning" wait.
        node = Node(node_id=node_id, resource_type=resource_type)
        
        # Use base pricing from config instead of mapped prices
        # to ensure consistency with the thesis document.
        pass
            
        node.state = NodeState.IDLE
        self.nodes[node_id] = node
        return node

    def get_available_nodes(self, required_memory: Optional[float] = None) -> List[Node]:
        """
        Get nodes that are currently idle or ready to be provisioned.
        
        In cloud mode, all IDLE nodes are candidates.
        """
        return [
            node for node in self.nodes.values()
            if node.state == NodeState.IDLE and node.can_accept_job(required_memory)
        ]

    def request_provisioning(self, node_id: int, current_time: float) -> float:
        """
        Request a GPU from RunPod.
        
        Returns the time (seconds) it will take for the GPU to be ready.
        This models "waiting for stock" based on qualitative High/Medium/Low status.
        """
        node = self.get_node(node_id)
        if not node:
            return 0.0

        # Update and get current status
        self.update_stock_statuses(current_time)
        status = self.get_stock_status(node.resource_type)
        
        # Determine delay based on status
        from cgrsp.data.runpod_availability import RunPodAvailabilityModel
        
        return self.availability_model.get_provisioning_delay(
            node.resource_type.value, current_time, node_id
        )

    def assign_job_to_node(self, node_id: int, job_id: int, current_time: float) -> bool:
        """
        Assign a job to a node, triggering cloud provisioning if necessary.
        """
        node = self.get_node(node_id)
        if not node or node.state != NodeState.IDLE:
            return False

        # In this simplified cloud model, we assume each assignment is a new rent request
        # To make it realistic, we could calculate the provisioning delay here.
        # However, the Simulator core handles the actual timing.
        # We just mark the node as RUNNING.
        
        return node.assign_job(job_id, current_time)

    def get_fleet_utilization(self, current_time: float) -> float:
        """Calculate utilization based on active rents."""
        total_slots = len(self.nodes)
        if total_slots == 0:
            return 0.0
        
        running = len(self.get_nodes_by_state(NodeState.RUNNING))
        return running / total_slots
