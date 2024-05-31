import argparse

from train import train_classifier, train_cyclegan

def main():
    parser = argparse.ArgumentParser(description="Pipeline for training classifier or CycleGAN")
    parser.add_argument("--model_type", type=str, choices=["classifier", "cycle-gan"], help="Type of model to train")
    parser.add_argument("--image_size", type=int, default=512, help="Size of input images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--resume_ckpt_path", type=str, default=None, help="Checkpoint path for resuming the training")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test data directory (for CycleGAN)")
    parser.add_argument("--project", type=str, default="CycleGAN-CounterFactual Explanation", help="Name of the project (for logging)")
    parser.add_argument("--job_name", type=str, default="training", help="Name of the training job (for logging)")
    parser.add_argument("--checkpoint_dir", type=str, default="./models", help="Directory to save model checkpoints")
    parser.add_argument("--classifier_path", type=str, default=None, help="Path to pre-trained classifier model (for CycleGAN)")

    args = parser.parse_args()

    if args.model_type == "classifier":
        train_classifier(args.image_size, args.batch_size, args.epochs, args.resume_ckpt_path, args.train_dir, args.val_dir, args.checkpoint_dir, args.project, args.job_name)
    elif args.model_type == "cycle-gan":
        if args.classifier_path == None:
            raise ValueError("Please provide the 'classifier checkpoint path' to train the cyle GAN model")
        train_cyclegan(args.image_size, args.batch_size, args.epochs, args.classifier_path, args.resume_ckpt_path, args.train_dir, args.val_dir, args.test_dir, args.checkpoint_dir, args.project, args.job_name)
    else:
        print("Invalid model type. Choose either 'classifier' or 'cycle-gan'.")

if __name__ == "__main__":
    main()
