variable "name" { type = string }
variable "cidr" { type = string }
variable "azs" { type = list(string) }

locals {
  private = [for i, _ in var.azs : cidrsubnet(var.cidr, 4, i)]
  public  = [for i, _ in var.azs : cidrsubnet(var.cidr, 8, 48 + i)]
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.0"

  name            = var.name
  cidr            = var.cidr
  azs             = var.azs
  private_subnets = local.private
  public_subnets  = local.public

  enable_nat_gateway   = true
  single_nat_gateway   = true # dev; one per AZ in production
  enable_dns_hostnames = true

  public_subnet_tags  = { "kubernetes.io/role/elb" = 1 }
  private_subnet_tags = { "kubernetes.io/role/internal-elb" = 1 }
}

output "vpc_id" { value = module.vpc.vpc_id }
output "private_subnets" { value = module.vpc.private_subnets }
output "vpc_cidr" { value = module.vpc.vpc_cidr_block }
