# Aborted attempt 1 (commit 64bf1d1, 2026-10-27 22:02-22:12)

Kept for the record, not used in any table or plot. The study process and the Ollama server
died together about 10 minutes after launch, after vLLM repeat 1 and four of Ollama's five
repeat-1 runs. No run reported an error. The `wsl.exe` client that started the study exited
with code 15.

Most likely cause: the study was launched through a background tool command that is stopped
after about 10 minutes, and WSL took the processes down with that client. Not ruled out: by
then llama-server's host-RAM prompt cache had grown to 8.1 GiB (`block1_ollama.log`) on a WSL
instance with 7.6 GB of RAM and 2 GB of swap, so memory pressure may have played a part. RAM
and swap were not recorded in this attempt.

What changed: the study now runs detached inside WSL (`scripts/run_study_detached.sh`),
records RAM and swap per run, and turns prompt caching off on both servers (see attempt 2).
It was re-run from scratch so that every reported run comes from one commit and one schedule.
