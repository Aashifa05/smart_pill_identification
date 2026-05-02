# Full training wrapper using frozen MobileNet backbone
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from train_feature_learning import PillClassifierTrainer, FeatureLearningModel, DataBalancer

if __name__ == '__main__':
    # Use the real dataset inside the repo
    dataset_dir = 'pill_project/Admin/media/train'
    output_dir = 'pill_project/Admin/media'
    os.makedirs(output_dir, exist_ok=True)

    trainer = PillClassifierTrainer(dataset_dir=dataset_dir,
                                    model_output_path=os.path.join(output_dir, 'model_feature_learning_final.keras'))

    images, labels, label_map = trainer.load_dataset()

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = DataBalancer.stratified_split(images, labels)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Class indices (label_map): {label_map}")

    # Build and show model summary
    model = FeatureLearningModel.build_model(len(label_map))
    model = FeatureLearningModel.compile_model(model, learning_rate=1e-4)
    print('\nModel summary:')
    model.summary()

    # Train up to 50 epochs with frozen backbone (trainer.train uses same callbacks)
    trained_model, history = trainer.train(epochs=50, batch_size=32, augmentation=True)

    # Final/Best metrics
    train_acc = history.history.get('accuracy', [None])[-1]
    val_acc = history.history.get('val_accuracy', [None])[-1]
    train_loss = history.history.get('loss', [None])[-1]
    val_loss = history.history.get('val_loss', [None])[-1]
    best_val_acc = max(history.history.get('val_accuracy', [0])) if history.history.get('val_accuracy') else None

    print(f"\nFinal training accuracy: {train_acc}")
    print(f"Final validation accuracy: {val_acc}")
    print(f"Best validation accuracy achieved: {best_val_acc}")
    print(f"Final training loss: {train_loss}")
    print(f"Final validation loss: {val_loss}")

    # Predictions on test set
    y_pred = trained_model.predict(X_test, verbose=0)
    y_test_labels = y_test.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)

    # Classification report
    report = classification_report(y_test_labels, y_pred_labels,
                                   target_names=[name for name, _ in sorted(label_map.items(), key=lambda x: x[1])])
    print('\nClassification Report:\n')
    print(report)

    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # Confusion matrix with professional formatting
    try:
        import seaborn as sns
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        
        # Get class names in order
        class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
        
        # Create figure with appropriate size for readability
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use seaborn heatmap for professional appearance
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        # Format labels and title
        ax.set_title('Confusion Matrix - Pill Classification', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_final.png'), dpi=200, bbox_inches='tight')
        print('Saved confusion_matrix_final.png')
        plt.close()
    except Exception as e:
        print('Failed to save confusion matrix:', e)

    # Accuracy and loss plots
    try:
        acc = history.history.get('accuracy', [])
        val_acc_hist = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss_hist = history.history.get('val_loss', [])

        # Accuracy plot with professional formatting
        epochs = range(1, len(acc) + 1)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(epochs, acc, 'b-', linewidth=2, label='Training Accuracy')
        ax1.plot(epochs, val_acc_hist, 'r-', linewidth=2, label='Validation Accuracy')
        ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_plot_final.png'), dpi=150, bbox_inches='tight')
        print('Saved accuracy_plot_final.png')
        plt.close()

        # Loss plot with professional formatting
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(epochs, loss, 'b-', linewidth=2, label='Training Loss')
        ax2.plot(epochs, val_loss_hist, 'r-', linewidth=2, label='Validation Loss')
        ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_plot_final.png'), dpi=150, bbox_inches='tight')
        print('Saved loss_plot_final.png')
        plt.close()
    except Exception as e:
        print('Failed to save accuracy/loss plots:', e)

    # Ensure final model file exists
    final_model_path = trainer.model_output_path
    print(f"Final model path: {final_model_path}")
    if os.path.exists(final_model_path):
        print('Final model saved successfully.')
    else:
        print('Final model not found; check training logs.')

    print('\nFull training run complete.')
