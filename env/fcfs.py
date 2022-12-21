from batsim_py.simulator import SimulatorHandler

class FCFSScheduler:
  def __init__(self, simulator: SimulatorHandler) -> None:
    self.simulator = simulator

  def __str__(self) -> str:
    return "FCFS"

  def schedule(self) -> None:
    """  First Come First Served policy """
    assert self.simulator.is_running

    for job in self.simulator.queue:
      available = self.simulator.platform.get_not_allocated_hosts()

      if job.res <= len(available):
        # Schedule if the job can start now.
        allocation = [h.id for h in available[:job.res]]
        self.simulator.allocate(job.id, allocation)
      else:
        # Otherwise, wait for resources.
        break

