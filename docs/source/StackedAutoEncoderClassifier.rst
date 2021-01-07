StackedAutoEncoderClassifier
============================

**Author:**

Rosefun, rosefun@foxmail.com

**Reference:**

None

::

    class semisupervised.StackedAutoEncoderClassifier(SAE, pretrain_epochs=100, finetune_epochs=100,
                     pretrain_optimizer_parameters=dict(lr=0.003, weight_decay=1e-5),
                     finetune_optimizer_parameters=dict(lr=0.003),
                     pretrain_batch_size=256, finetune_batch_size=256, pretrain_optimizer=None,
                     finetune_optimizer=None, patience=40,
                     device_name="auto", verbose=1, save_pretrain_model=False)

**Parameters:**

-  **SAE**

SAE model written by Pytorch.

-  **pretrain\_epochs**

Parameter for SAE.

-  **finetune\_epochs**

Parameter for SAE.

-  **pretrain\_optimizer\_parameters**

Parameter for SAE optimizer.

-  **finetune\_optimizer\_parameters**

Parameter for SAE optimizer.

-  **pretrain\_batch\_size**

Batch size in pretrain stage.

-  **finetune\_batch\_size**

Batch size in finetune stage.

-  **pretrain\_optimizer**

Optimizer in pretrain stage.

-  **finetune\_optimizer**

Optimizer in finetune stage.

-  **patience**

Parameter for early stopping.

-  **device\_name**

"Cuda" or "Cpu".

-  **verbose**

Int type. ``0`` means nothing to show. ``n`` means show each model
training information in each ``n`` epochs.

-  **save\_pretrain\_model**

``True`` or ``False``. Whether save the parameters of pretrain model.

**Methods:**

-  **save\_model**

**Parameters:**

​ None

**Return:**

​ Checkpoint file of model.

-  **create\_dataloader**\ (X, y=None, batch\_size=256, shuffle=False,
   device="cuda")

**Parameters:**

​ ``X`` : numpy array or pandas DataFrame or Tensor type.

**Return:**

​ DataLoader.

-  **fit(X, y, is\_pretrain=True, validation\_data=None)**

**Parameters:**

​ ``X`` : 2-dim numpy array.

​ ``y`` : 1-dim numpy array, scalar value. If value equals ``-1``, it
means unlabeled sample.

**Return:**

​ SAE model.

-  **predict\_epoch**\ (valid\_loader)

**Parameters:**

​ ``valid_loader`` : DataLoader.

**Return:**

​ Loss of input dataloader.

-  **predict**\ (X\_test)

**Parameters:**

​ ``X_test`` : 2-dim numpy array.

**Return**:

​ 1-dim numpy array, predicted result.

-  **predict\_proba**\ (X\_test)

**Parameters**:

​ ``X_test`` :2-dim numpy array.

**Return**:

​ 2-dim numpy array, predicted result.

-  **score**\ (X, y, sample\_weight=None)

**Parameters**

​ ``X`` : 2-dim numpy array.

​ ``y`` : 1-dim numpy array.

**Return:**

​ Accuracy score of (X, y).
