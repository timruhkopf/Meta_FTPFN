When Rsync fails:

```text
 Download '/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/asdf_concat_smoothing' to '/home/ruhkopf/VSCode/Meta_FTPFN/outputs/asdf_concat_smoothing' using rsync
[30.03.26, 09:32] /usr/bin/rsync -zar -e "ssh -p 22 " --exclude=.svn --exclude=.cvs --exclude=.idea --exclude=.DS_Store --exclude=.git --exclude=.hg --exclude=*.hprof --exclude=*.pyc ruhkopf@ulysses:/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/asdf_concat_smoothing/ asdf_concat_smoothing
[30.03.26, 09:32] ruhkopf@130.75.145.188's password:
[30.03.26, 09:32] Permission denied, please try again.
[30.03.26, 09:32] ruhkopf@130.75.145.188's password:
[30.03.26, 09:32] Permission denied, please try again.
[30.03.26, 09:32] ruhkopf@130.75.145.188's password:
[30.03.26, 09:32] ruhkopf@130.75.145.188: Permission denied (publickey,password).
[30.03.26, 09:32] rsync: connection unexpectedly closed (0 bytes received so far) [Receiver]
[30.03.26, 09:32] rsync error: unexplained error (code 255) at io.c(235) [Receiver=3.1.3]
[30.03.26, 09:32] Failed to transfer folder '/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/asdf_concat_smoothing'. Unknown message with code "Rsync failed with exit code: 255".

```

then add the identity to the currently running (and configured in ssh config) ssh-agent:

```bash
ssh-add ~/.ssh/ulysses
```

this is only a temporary fix until we reboot. 

An alternative resolution might be removing the known_hosts file content and accept new fingerprints. 