import torch.nn as nn
import numpy as np
import torch


def create_mask_tasks(unit_count, task_count, alpha, beta):
    _unit_mapping = torch.zeros((task_count, unit_count))
    num_filled_channel = int(unit_count * beta)
    last_index = unit_count - num_filled_channel
    index = ((np.linspace(0, 1, task_count) ** alpha) * last_index).round().astype(np.int64)
    for task in range(task_count):
        _unit_mapping[task, index[task] : index[task] + num_filled_channel] = torch.ones((num_filled_channel))

    return _unit_mapping


class TaskRouter(nn.Module):

    """
    Same as https://github.com/gstrezoski/TaskRouting/blob/master/taskrouting.py
    Applies task specific masking out individual units in a layer.

    Args:
    unit_count  (int): Number of input channels going into the Task Routing layer.
    task_count  (int): Number of tasks. (IN STL it applies to number of output classes)
    sigma (int): Ratio for routed units per task.
    """

    def __init__(self, unit_count, taskcount, alpha, beta):
        super(TaskRouter, self).__init__()
        self.unit_count = unit_count
        # Just initilize it with 0. This gets changed right after the model is loaded so the value is never used.
        # We store the active mask for the Task Routing Layer here.
        self.active_task = 0

        _unit_mapping = create_mask_tasks(
            unit_count,
            taskcount,
            alpha=alpha,
            beta=beta,
        )
        self.register_buffer("_unit_mapping", _unit_mapping)

    def set_active_task(self, active_task):
        self.active_task = active_task
        self.active_mask = torch.index_select(self._unit_mapping, 0, (torch.ones(len(active_task), device=active_task.device) * self.active_task ** 999).long()).unsqueeze(1)
        return active_task

    def forward(self, input, module_input):
        input = module_input + input * (1 - self.active_mask)
        return input