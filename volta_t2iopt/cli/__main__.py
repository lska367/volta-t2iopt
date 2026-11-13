"""`python -m volta_t2iopt.cli ...` 的入口（matrix 子命令靠它起子进程）。"""

from . import main

raise SystemExit(main())
