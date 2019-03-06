
if [ $1 == "-f" ]; then
    rm ./logs/ddpg/*
    rm ./ckpt/ddpg/*
else
    python ddpg.py
fi

# rm ./logs/ddpg/*
# rm ./ckpt/ddpg/*


