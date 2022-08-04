FROM gitpod/workspace-full

# Install custom tools, runtime, etc.
RUN sudo usermod -a -G video gitpod

