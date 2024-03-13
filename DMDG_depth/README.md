# Applying General mDG objective to depth estimation task. 

The code is modified from # [VA-DepthNet: A Variational Approach to Single Image Depth Prediction](https://openreview.net/forum?id=xjxUjHa_Wpa) 
THe dependencies and datasets you can follow VA-DepthNet at [here](https://github.com/cnexah/VA-DepthNet)

## Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python vadepthnet/train.py configs/arguments_train_nyu.txt
```


## Evaluation
Evaluate the NYUv2 model:
```
python vadepthent/eval.py configs/arguments_eval_nyu.txt
```

## important notes
- YOU NEED TO MERGE THE TRAIN AND TEST SET TOGETHER in NYU dataset for conduct the mult-domain generalization task. 
We split the merged NYU dataset into four domains:
```
self.env_class = {
            'school':
                ['study_room',
                 'study',
                 'student_lounge',
                 'printer_room',
                 'computer_lab',
                 'classroom', ],
            'office':
                ['reception_room',
                 'office_kitchen',
                 'office',
                 'nyu_office',
                 'conference_room', ],
            'home':
                ['playroom',
                 'living_room',
                 'laundry_room',
                 'kitchen',
                 'indoor_balcony',
                 'home_storage',
                 'home_office',
                 'foyer',
                 'dining_room',
                 'dinette',
                 'bookstore',
                 'bedroom',
                 'bathroom',
                 'basement'],
            'commercial':
                ['furniture_store',
                 'excercise_room',
                 'cafe', ]
        }
```
You can see the home domain contains the most categories and it has the largest domain shift. 
Therefore, the most difficult situation for generalization is when the home domain is held out for testing.

- You need to specify which domain you want to left for tesing in the config file (--te_idx 0).
- The y_mapping is embeded in the vadepthnet.py file, please ignore the one in the train file. 
Of course you can define you own y_mapping :p
- We tried to filp H(phi(X)|psi(Y)) - H(phi(X)) to H(psi(Y)|phi(X)) - H(psi(Y)), 
the network collapsed totally. If you are interested you can also have a test.
