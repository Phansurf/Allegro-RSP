from nequip.train import Trainer
from nequip.utils._global_options import _set_global_options

material = "HfO2"
model = "3-step"
pth_path = f"results/reciprocal/{material}/{material}({model})/trainer.pth"

_set_global_options(dict(default_dtype="float64"))
trainer=Trainer.from_file(pth_path)

# early_stopping_patiences.validation_loss
# trainer.kwargs["early_stopping_patiences"]["validation_loss"] = 1000
# print("early_stopping_patiences.validation_loss:", trainer.kwargs["early_stopping_patiences"]["validation_loss"])
# print("trainer.early_stopping_conds.patiences:", trainer.early_stopping_conds.patiences, "-> change at next load")

# batch_size
# trainer.batch_size = 64
# trainer.validation_batch_size = 64
# print("batch_size:", trainer.batch_size)
# print("validation_batch_size:", trainer.validation_batch_size)

# dataloader_num_workers & dataloader_prefetch_factor
# trainer.dataloader_num_workers = 2
# trainer.dataloader_prefetch_factor = 2
# print("dataloader_num_workers:", trainer.dataloader_num_workers)
# print("dataloader_prefetch_factor:", trainer.dataloader_prefetch_factor)

# stage
# print("stage:", trainer.stage)
# trainer.stage=2
# trainer.init()
# print("stage:", trainer.stage)

# max_epochs
# trainer.max_epochs = 100000
# print("max_epochs:", trainer.max_epochs)

# lr_scheduler_patience
print("lr_scheduler_patience:", trainer.kwargs["lr_scheduler_patience"])
new_patience = 50
trainer.kwargs["lr_scheduler_patience"] = new_patience
trainer.lr_scheduler_kwargs["patience"] = new_patience
trainer.lr_sched.patience = new_patience
print("lr_scheduler_patience:", trainer.lr_sched.patience)

trainer.save(pth_path)
