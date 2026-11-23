variable "name" { type = string }
variable "kubernetes_version" { type = string }
variable "vpc_id" { type = string }
variable "subnet_ids" { type = list(string) }
variable "api_instance_types" { type = list(string) }
variable "spark_instance_types" { type = list(string) }
variable "spark_max_nodes" { type = number }

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = var.name
  kubernetes_version = var.kubernetes_version
  vpc_id             = var.vpc_id
  subnet_ids         = var.subnet_ids

  endpoint_public_access                   = true
  enable_cluster_creator_admin_permissions = true
  enable_irsa                              = true

  addons = {
    coredns                = {}
    kube-proxy             = {}
    vpc-cni                = { before_compute = true }
    eks-pod-identity-agent = { before_compute = true }
  }

  eks_managed_node_groups = {
    # API, console, analyst, Airflow
    serving = {
      instance_types = var.api_instance_types
      min_size       = 2
      max_size       = 10
      desired_size   = 3
      labels         = { workload = "serving" }
    }
    # Spark executors for the weekly scoring; scales to zero between runs
    spark = {
      instance_types = var.spark_instance_types
      capacity_type  = "SPOT"
      min_size       = 0
      max_size       = var.spark_max_nodes
      desired_size   = 0
      labels         = { workload = "spark" }
      taints = {
        spark = { key = "workload", value = "spark", effect = "NO_SCHEDULE" }
      }
    }
  }
}

output "cluster_name" { value = module.eks.cluster_name }
output "oidc_provider_arn" { value = module.eks.oidc_provider_arn }
output "cluster_security_group_id" { value = module.eks.node_security_group_id }
