FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

RUN echo 'root:root' | chpasswd

RUN apt update && \
    apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    apt clean

RUN echo "session optional pam_loginuid.so" >> /etc/pam.d/sshd

ENV NOTVISIBLE=in_users_profile
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /root

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
# nvidia-smi
