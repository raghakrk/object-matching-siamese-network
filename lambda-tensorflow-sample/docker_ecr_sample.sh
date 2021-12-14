# Build the docker image
docker build -t  lambda-tensorflow-sample .

# Create a ECR repository
aws ecr create-repository --repository-name lambda-tensorflow-sample --image-scanning-configuration scanOnPush=true --region us-west-2

# Tag the image to match the repository name
docker tag lambda-tensorflow-sample:latest 465906353389.dkr.ecr.us-west-2.amazonaws.com/lambda-tensorflow-sample:latest

# Register docker to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 465906353389.dkr.ecr.us-west-2.amazonaws.com

# Push the image to ECR
docker push 465906353389.dkr.ecr.us-west-2.amazonaws.com/lambda-tensorflow-sample:latest