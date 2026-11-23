variable "name" { type = string }
variable "oidc_provider_arn" { type = string }
variable "namespace" { type = string }
variable "bucket_arns" { type = list(string) }

# IAM roles for service accounts (IRSA): jobs read/write the lake and artifacts,
# the API only reads artifacts.
module "jobs_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts"
  version = "~> 6.0"

  name = "${var.name}-jobs"
  oidc_providers = {
    main = {
      provider_arn               = var.oidc_provider_arn
      namespace_service_accounts = ["${var.namespace}:ope-jobs"]
    }
  }
  policies = { s3 = aws_iam_policy.jobs.arn }
}

module "api_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts"
  version = "~> 6.0"

  name = "${var.name}-api"
  oidc_providers = {
    main = {
      provider_arn               = var.oidc_provider_arn
      namespace_service_accounts = ["${var.namespace}:default"]
    }
  }
  policies = { s3 = aws_iam_policy.api.arn }
}

resource "aws_iam_policy" "jobs" {
  name = "${var.name}-jobs-s3"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
      Resource = flatten([for a in var.bucket_arns : [a, "${a}/*"]])
    }]
  })
}

resource "aws_iam_policy" "api" {
  name = "${var.name}-api-s3-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = flatten([for a in var.bucket_arns : [a, "${a}/*"]])
    }]
  })
}

output "jobs_role_arn" { value = module.jobs_role.arn }
output "api_role_arn" { value = module.api_role.arn }
