---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: MaxStrange

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Logs**
If applicable, add logs from the device (SSH over to your device before running these commands):

* `mkdir logs`
* `cat /etc/os-release > logs/os-release.txt`
* `cat /etc/os-subrelease > logs/os-subrelease.txt`
* `cat /etc/adu-version > logs/adu-version.txt`
* `sudo iotedge logs azureeyemodule > logs/azureeyemodule.txt`
* `tar -zcvf logs.tar.gz logs`

Now, from your PC, copy the logs back over to your PC with (this will work on Windows PowerShell/CMD, Mac, or Linux):

`scp [remote username]@[IP address]:~/logs.tar.gz logs.tar.gz`

`[remote username]` is the SSH username chosen during the OOBE setup process. If you did not set up an SSH login during the OOBE, your remote username is root.

**Additional context**
Add any other context about the problem here.
