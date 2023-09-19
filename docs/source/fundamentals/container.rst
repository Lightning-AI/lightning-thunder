Get Thunder (NVIDIA only)
#########################

*This section is intended for team members at NVIDIA only.*

For team members at NVIDIA, *thunder* is available in a pre-built container.
If you don't qualify or need to install *thunder* from scratch, follow the :doc:`Install <installation>` guide.

To get *thunder* up and running in seconds use the following prebuilt container::

  gitlab-master.nvidia.com:5005/dl/pytorch/update-scripts:pjnl-latest

This container already has *thunder* and multiple executors configured ready for you to use. If you experience any issues with this container please report so in our shared slack channel mentioned in the points of contact section above.

pjnl-latest can be replaced with a date in YYYYMMDD format. For example::

  gitlab-master.nvidia.com:5005/dl/pytorch/update-scripts:pjnl-20230909

specifies the container from September 9th, 2023.

The source for lightning.compile and nvFuser can be found in the directory `/opt/pytorch`.
