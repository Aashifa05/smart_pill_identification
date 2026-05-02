# Dry-run wrapper to validate training pipeline on real Kaggle dataset
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from train_feature_learning import PillClassifierTrainer, FeatureLearningModel, DataBalancer

if __name__ == '__main__':
    # Use pill_project/Admin/media/train which contains the real Kaggle-derived images
    trainer = PillClassifierTrainer(dataset_dir='pill_project/Admin/media/train',
                                    model_output_path='pill_project/Admin/media/model_feature_learning_dry.keras')

    # Load dataset
    images, labels, label_map = trainer.load_dataset()

    # Split (full datasets, stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(images, labels)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Class indices (label_map): {label_map}")

    # Build and show model summary
    model = FeatureLearningModel.build_model(len(label_map))
    model = FeatureLearningModel.compile_model(model)
    print('\nModel summary:')
    model.summary()

    # Train for 2 epochs (dry-run)
    model, history = trainer.train(epochs=2, batch_size=32, augmentation=True)

    # Print final train and val accuracy from history
    train_acc = history.history.get('accuracy', [None])[-1]
    val_acc = history.history.get('val_accuracy', [None])[-1]
    train_loss = history.history.get('loss', [None])[-1]
    val_loss = history.history.get('val_loss', [None])[-1]

    print(f"\nFinal training accuracy: {train_acc}")
    print(f"Final validation accuracy: {val_acc}")
    print(f"Final training loss: {train_loss}")
    print(f"Final validation loss: {val_loss}")

    # Evaluate and create confusion matrix
    y_pred = model.predict(X_test, verbose=0)
    y_test_labels = y_test.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)

    # Classification report
    report = classification_report(y_test_labels, y_pred_labels,
                                   target_names=[name for name, _ in sorted(label_map.items(), key=lambda x: x[1])])
    print('\nClassification Report:\n')
    print(report)

    # Confusion matrix plot
    try:
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print('Saved confusion_matrix.png')
    except Exception as e:
        print('Failed to save confusion matrix:', e)

    # Accuracy and loss plots
    try:
        acc = history.history.get('accuracy', [])
        val_acc_hist = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss_hist = history.history.get('val_loss', [])

        # Accuracy plot
        plt.figure()
        plt.plot(acc, label='train_accuracy')
        plt.plot(val_acc_hist, label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('accuracy_plot.png', dpi=150)
        print('Saved accuracy_plot.png')

        # Loss plot
        plt.figure()
        plt.plot(loss, label='train_loss')
        plt.plot(val_loss_hist, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_plot.png', dpi=150)
        print('Saved loss_plot.png')
    except Exception as e:
        print('Failed to save accuracy/loss plots:', e)

    print('\nDry-run complete.')
