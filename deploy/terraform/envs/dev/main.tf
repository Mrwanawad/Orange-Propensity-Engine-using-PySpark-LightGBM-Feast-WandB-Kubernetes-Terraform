# Dev environment. `terraform validate` and `plan` only: nothing here has been applied.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
  # backend "s3" { bucket = "ope-tfstate", key = "dev/terraform.tfstate", region = "eu-west-1", use_lockfile = true }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "ope", env = "dev", managed_by = "terraform" } }
}

variable "region" {
  type    = string
  default = "eu-west-1"
}

locals {
  name = "ope-dev"
  azs  = ["${var.region}a", "${var.region}b", "${var.region}c"]
}

module "network" {
  source = "../../modules/network"
  name   = local.name
  cidr   = "10.40.0.0/16"
  azs    = local.azs
}

module "eks" {
  source               = "../../modules/eks"
  name                 = local.name
  kubernetes_version   = "1.33"
  vpc_id               = module.network.vpc_id
  subnet_ids           = module.network.private_subnets
  api_instance_types   = ["m7i.xlarge"]
  spark_instance_types = ["c7i.4xlarge", "c6i.4xlarge"]
  spark_max_nodes      = 20
}

module "s3" {
  source  = "../../modules/s3"
  prefix  = local.name
  buckets = ["lake", "artifacts", "raw"]
}

module "ecr" {
  source       = "../../modules/ecr"
  repositories = ["ope-api", "ope-jobs"]
}

module "redis" {
  source                    = "../../modules/redis"
  name                      = local.name
  vpc_id                    = module.network.vpc_id
  subnet_ids                = module.network.private_subnets
  allowed_security_group_id = module.eks.cluster_security_group_id
  node_type                 = "cache.r7g.large"
  shards                    = 2
  replicas_per_shard        = 1
}

module "iam" {
  source            = "../../modules/iam"
  name              = local.name
  oidc_provider_arn = module.eks.oidc_provider_arn
  namespace         = "ope"
  bucket_arns       = values(module.s3.bucket_arns)
}

output "cluster_name" { value = module.eks.cluster_name }
output "ecr" { value = module.ecr.repository_urls }
output "redis_endpoint" { value = module.redis.endpoint }
output "jobs_role_arn" { value = module.iam.jobs_role_arn }
output "buckets" { value = module.s3.bucket_names }
