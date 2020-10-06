# !/bin/sh

source ~/env/bin/activate
pip install -r requirements.txt
chmod +x main_cl.py main_pretrain.py compare_both.py
./compare_both.py --lambda-500=10000 --lambda=10000 --o-lambda-500=10000 --o-lambda=1000 --c-500=10000 --c=100 --n-seeds=4 --reinit --slot=4 --shift=4
git add ./store/results
git commit -m "res for slot 4 shift 4"
git push
sudo shutdown now
