#!/bin/bash
# Deploy frontend to S3 and CloudFront

set -e

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "=== Loading environment from .env ==="
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
else
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and configure your domains"
    exit 1
fi

# Validate required environment variables
if [ -z "$BACKEND_DOMAIN" ]; then
    echo "ERROR: BACKEND_DOMAIN not set in .env"
    exit 1
fi

if [ -z "$FRONTEND_DOMAIN" ]; then
    echo "ERROR: FRONTEND_DOMAIN not set in .env"
    exit 1
fi

if [ -z "$FRONTEND_S3_BUCKET" ]; then
    echo "ERROR: FRONTEND_S3_BUCKET not set in .env"
    exit 1
fi

REGION="${AWS_REGION:-ap-southeast-2}"

echo "=== Configuration ==="
echo "Frontend domain: $FRONTEND_DOMAIN"
echo "Backend domain: $BACKEND_DOMAIN"
echo "S3 bucket: $FRONTEND_S3_BUCKET"
echo "Region: $REGION"

echo "=== Building frontend ==="
mkdir -p dist/frontend

# Copy the index.html (single-file Vue app)
cp src/memory/web/templates/index.html dist/frontend/index.html

# Update API endpoint to use the backend domain from env
# Using gsed on macOS or sed on Linux
if command -v gsed &> /dev/null; then
    SED_CMD="gsed"
else
    SED_CMD="sed"
fi

# Replace fetch('/api/ with fetch('BACKEND_DOMAIN/api/
$SED_CMD -i "s|fetch('/api/|fetch('${BACKEND_DOMAIN}/api/|g" dist/frontend/index.html
# Replace fetch("/api/ with fetch("BACKEND_DOMAIN/api/
$SED_CMD -i "s|fetch(\"/api/|fetch(\"${BACKEND_DOMAIN}/api/|g" dist/frontend/index.html
# Replace fetch(\`/api/ with fetch(\`BACKEND_DOMAIN/api/ (template strings)
$SED_CMD -i 's|fetch(`/api/|fetch(`'"${BACKEND_DOMAIN}"'/api/|g' dist/frontend/index.html
# Replace url = \`/api/ with url = \`BACKEND_DOMAIN/api/ (dynamic URL variables)
$SED_CMD -i 's|= `/api/|= `'"${BACKEND_DOMAIN}"'/api/|g' dist/frontend/index.html
# Replace url = '/api/ with url = 'BACKEND_DOMAIN/api/
$SED_CMD -i "s|= '/api/|= '${BACKEND_DOMAIN}/api/|g" dist/frontend/index.html

echo "=== Verifying URL replacement ==="
grep -c "fetch('${BACKEND_DOMAIN}" dist/frontend/index.html || true
grep -c 'fetch(`'"${BACKEND_DOMAIN}" dist/frontend/index.html || true
grep -c "= '${BACKEND_DOMAIN}" dist/frontend/index.html || true

echo "=== Uploading to S3 ==="
aws s3 sync dist/frontend/ s3://${FRONTEND_S3_BUCKET}/ --delete --region $REGION

echo "=== Invalidating CloudFront cache ==="
# Extract domain without https://
FRONTEND_HOST=$(echo "$FRONTEND_DOMAIN" | sed 's|https://||' | sed 's|http://||')
DIST_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?Aliases.Items[?@=='${FRONTEND_HOST}']].Id" --output text)
if [ -n "$DIST_ID" ] && [ "$DIST_ID" != "None" ]; then
    aws cloudfront create-invalidation --distribution-id $DIST_ID --paths "/*"
    echo "Cache invalidation created for distribution $DIST_ID"
else
    echo "No CloudFront distribution found for $FRONTEND_HOST"
fi

echo "=== Frontend deployed! ==="
