
if [ $1 == "-f" ]; then
    rm ./logs/maddpg/*
    rm ./ckpt/maddpg/*
else
    python maddpg.py
fi

# rm ./logs/ddpg/*
# rm ./ckpt/ddpg/*


