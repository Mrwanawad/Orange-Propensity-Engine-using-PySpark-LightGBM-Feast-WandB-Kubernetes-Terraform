variable "prefix" { type = string }
variable "buckets" { type = list(string) }

resource "aws_s3_bucket" "this" {
  for_each = toset(var.buckets)
  bucket   = "${var.prefix}-${each.key}"
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "bucket_arns" { value = { for k, b in aws_s3_bucket.this : k => b.arn } }
output "bucket_names" { value = { for k, b in aws_s3_bucket.this : k => b.bucket } }
