nengo code from "How to build brain".

to install nengo, use
sudo pip3 install nengo nengo-gui

To run x.py,  use nengo --no-browser x.py

This create http server.  If you have server running on non client machines, you would need to use haproxy or something similar.  Use haproxy.cfg in the sources.

Running with python x.py will show the static plots.
