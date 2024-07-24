# Download Guide
## Install AWS CLI
Follow the instructions below to install the AWS CLI on your machine.

### Windows

1. Download the AWS CLI Installer:
   - Visit the [official AWS CLI download page](https://aws.amazon.com/cli/).
   - Click on the latest Windows installer link to download.

2. Run the Installer:
   - Locate the downloaded `.msi` file and double-click to run the installer.
   - Follow the installation prompts.

3. Confirm Installation:
   Open Command Prompt and run the following command:
   ```sh
   aws --version
   ```
   You should see output similar to: `aws-cli/2.x.x Python/3.x.x Windows/10` or later.

### macOS

1. Using Homebrew:
   ```sh
   brew install awscli
   ```

2. Confirm Installation:
   Open Terminal and run the following command:
   ```sh
   aws --version
   ```
   You should see output similar to: `aws-cli/2.x.x Python/3.x.x Darwin/20.x.x` or later.

### Linux

1. Using Package Manager (e.g., apt for Ubuntu/Debian):
   ```sh
   sudo apt update
   sudo apt install awscli -y
   ```

2. Confirm Installation:
   Open Terminal and run the following command:
   ```sh
   aws --version
   ```
   You should see output similar to: `aws-cli/2.x.x Python/3.x.x Linux/4.x.x` or later.

## Sync S3 Bucket to Local Machine

Since you are using the `--no-sign-request` option, you do not need to configure the AWS CLI with your credentials.

1. Open a terminal or Command Prompt.
2. Run the following command to sync the S3 bucket to a local directory:
   ```sh
   aws s3 sync --no-sign-request s3://lasa-2024/ ./lasa-2024
   ```
   This command will copy all files and folders from the `lasa-2024` S3 bucket to a local directory named `lasa-2024`.