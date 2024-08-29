_base_ = [
    '../_base_/models/stdc.py', '../_base_/datasets/stdc1_cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# param_scheduler = [
#     dict(
#         type='ReduceOnPlateauLR',
#         monitor='mIoU',
#         rule='greater',  # depending on the metric (less for loss, greater for accuracy/mIoU)
#         factor=0.1,  # Factor by which the learning rate will be reduced
#         patience=10,  # Number of epochs with no improvement after which learning rate will be reduced
#         threshold=0.01,  # Threshold for measuring improvement
#         cooldown=0,  # Number of epochs to wait before resuming normal operation
#         min_value=1e-6,  # Minimum learning rate
#         verbose=False
#     )
# ]
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=80000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=2, num_workers=4) #12
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
