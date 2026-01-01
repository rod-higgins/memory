"""
Cloud service data source adapters.

Ingest metadata and content from cloud providers:
- AWS (via CLI)
- Future: GCP, Azure
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from .base import DataCategory, DataPoint, DataSource


class AWSSource(DataSource):
    """
    Ingest AWS resource metadata and logs via AWS CLI.

    Captures:
    - S3 bucket listings and object metadata
    - Lambda function configurations
    - EC2 instance metadata
    - DynamoDB table info
    - CloudWatch log groups
    - IAM users and roles
    """

    def __init__(
        self,
        profile: str = "default",
        region: str | None = None,
        include_s3: bool = True,
        include_lambda: bool = True,
        include_ec2: bool = True,
        include_dynamodb: bool = True,
        include_cloudwatch: bool = True,
        include_iam: bool = True,
    ):
        self._profile = profile
        self._region = region
        self._include_s3 = include_s3
        self._include_lambda = include_lambda
        self._include_ec2 = include_ec2
        self._include_dynamodb = include_dynamodb
        self._include_cloudwatch = include_cloudwatch
        self._include_iam = include_iam
        self._cli_available: bool | None = None

    @property
    def name(self) -> str:
        return "AWS Cloud"

    @property
    def category(self) -> DataCategory:
        return DataCategory.PROJECT

    @property
    def description(self) -> str:
        return "AWS resources and infrastructure metadata"

    def _run_aws_command(self, args: list[str]) -> dict[str, Any] | list[Any] | None:
        """Run an AWS CLI command and return parsed JSON output."""
        cmd = ["aws"]
        if self._profile:
            cmd.extend(["--profile", self._profile])
        if self._region:
            cmd.extend(["--region", self._region])
        cmd.extend(args)
        cmd.extend(["--output", "json"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass
        return None

    async def is_available(self) -> bool:
        """Check if AWS CLI is configured and accessible."""
        if self._cli_available is not None:
            return self._cli_available

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["sts", "get-caller-identity"])
        )
        self._cli_available = result is not None
        return self._cli_available

    async def estimate_count(self) -> int:
        """Estimate number of AWS resources."""
        if not await self.is_available():
            return 0

        # Rough estimate based on enabled services
        count = 0
        loop = asyncio.get_event_loop()

        if self._include_s3:
            buckets = await loop.run_in_executor(
                None, lambda: self._run_aws_command(["s3api", "list-buckets"])
            )
            if buckets and "Buckets" in buckets:
                count += len(buckets["Buckets"]) * 10  # Estimate objects per bucket

        if self._include_lambda:
            functions = await loop.run_in_executor(
                None, lambda: self._run_aws_command(["lambda", "list-functions"])
            )
            if functions and "Functions" in functions:
                count += len(functions["Functions"])

        return count

    async def iterate(self) -> AsyncIterator[DataPoint]:
        """Iterate through AWS resources."""
        if not await self.is_available():
            return

        loop = asyncio.get_event_loop()

        # Get caller identity for context
        identity = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["sts", "get-caller-identity"])
        )
        account_id = identity.get("Account", "unknown") if identity else "unknown"

        # S3 Buckets
        if self._include_s3:
            async for dp in self._iterate_s3(loop, account_id):
                yield dp

        # Lambda Functions
        if self._include_lambda:
            async for dp in self._iterate_lambda(loop, account_id):
                yield dp

        # EC2 Instances
        if self._include_ec2:
            async for dp in self._iterate_ec2(loop, account_id):
                yield dp

        # DynamoDB Tables
        if self._include_dynamodb:
            async for dp in self._iterate_dynamodb(loop, account_id):
                yield dp

        # CloudWatch Log Groups
        if self._include_cloudwatch:
            async for dp in self._iterate_cloudwatch(loop, account_id):
                yield dp

        # IAM Users and Roles
        if self._include_iam:
            async for dp in self._iterate_iam(loop, account_id):
                yield dp

    async def _iterate_s3(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through S3 buckets."""
        buckets = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["s3api", "list-buckets"])
        )

        if not buckets or "Buckets" not in buckets:
            return

        for bucket in buckets["Buckets"]:
            bucket_name = bucket.get("Name", "")
            creation_date = bucket.get("CreationDate", "")

            content = f"S3 Bucket: {bucket_name}\nCreated: {creation_date}\nAccount: {account_id}"

            yield DataPoint(
                content=content,
                category=DataCategory.PROJECT,
                subcategory="s3_bucket",
                source_type="aws_s3",
                source_id=bucket_name,
                original_date=self._parse_aws_date(creation_date),
                topics=["aws", "s3", "storage", "cloud"],
                raw_data=bucket,
            )

    async def _iterate_lambda(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through Lambda functions."""
        functions = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["lambda", "list-functions"])
        )

        if not functions or "Functions" not in functions:
            return

        for func in functions["Functions"]:
            func_name = func.get("FunctionName", "")
            runtime = func.get("Runtime", "")
            description = func.get("Description", "")
            last_modified = func.get("LastModified", "")
            memory = func.get("MemorySize", 0)
            timeout = func.get("Timeout", 0)

            content = (
                f"Lambda Function: {func_name}\n"
                f"Runtime: {runtime}\n"
                f"Description: {description}\n"
                f"Memory: {memory}MB, Timeout: {timeout}s\n"
                f"Last Modified: {last_modified}"
            )

            yield DataPoint(
                content=content,
                category=DataCategory.CODE,
                subcategory="lambda_function",
                source_type="aws_lambda",
                source_id=func.get("FunctionArn", func_name),
                original_date=self._parse_aws_date(last_modified),
                topics=["aws", "lambda", "serverless", runtime.lower() if runtime else ""],
                raw_data=func,
            )

    async def _iterate_ec2(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through EC2 instances."""
        reservations = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["ec2", "describe-instances"])
        )

        if not reservations or "Reservations" not in reservations:
            return

        for reservation in reservations["Reservations"]:
            for instance in reservation.get("Instances", []):
                instance_id = instance.get("InstanceId", "")
                instance_type = instance.get("InstanceType", "")
                state = instance.get("State", {}).get("Name", "")
                launch_time = instance.get("LaunchTime", "")

                # Get name tag
                name = ""
                for tag in instance.get("Tags", []):
                    if tag.get("Key") == "Name":
                        name = tag.get("Value", "")
                        break

                content = (
                    f"EC2 Instance: {name or instance_id}\n"
                    f"ID: {instance_id}\n"
                    f"Type: {instance_type}\n"
                    f"State: {state}\n"
                    f"Launched: {launch_time}"
                )

                yield DataPoint(
                    content=content,
                    category=DataCategory.PROJECT,
                    subcategory="ec2_instance",
                    source_type="aws_ec2",
                    source_id=instance_id,
                    original_date=self._parse_aws_date(launch_time),
                    topics=["aws", "ec2", "compute", "infrastructure"],
                    raw_data=instance,
                )

    async def _iterate_dynamodb(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through DynamoDB tables."""
        tables = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["dynamodb", "list-tables"])
        )

        if not tables or "TableNames" not in tables:
            return

        for table_name in tables["TableNames"]:
            # Get table details
            table_info = await loop.run_in_executor(
                None,
                lambda tn=table_name: self._run_aws_command(
                    ["dynamodb", "describe-table", "--table-name", tn]
                ),
            )

            if table_info and "Table" in table_info:
                table = table_info["Table"]
                creation_time = table.get("CreationDateTime", "")
                item_count = table.get("ItemCount", 0)
                size_bytes = table.get("TableSizeBytes", 0)

                content = (
                    f"DynamoDB Table: {table_name}\n"
                    f"Items: {item_count}\n"
                    f"Size: {size_bytes / 1024 / 1024:.2f} MB\n"
                    f"Created: {creation_time}"
                )

                yield DataPoint(
                    content=content,
                    category=DataCategory.PROJECT,
                    subcategory="dynamodb_table",
                    source_type="aws_dynamodb",
                    source_id=table.get("TableArn", table_name),
                    original_date=self._parse_aws_date(creation_time),
                    topics=["aws", "dynamodb", "database", "nosql"],
                    raw_data=table,
                )

    async def _iterate_cloudwatch(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through CloudWatch log groups."""
        log_groups = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["logs", "describe-log-groups"])
        )

        if not log_groups or "logGroups" not in log_groups:
            return

        for group in log_groups["logGroups"]:
            group_name = group.get("logGroupName", "")
            creation_time = group.get("creationTime", 0)
            stored_bytes = group.get("storedBytes", 0)

            # Convert timestamp
            created_dt = None
            if creation_time:
                created_dt = datetime.fromtimestamp(creation_time / 1000)

            content = (
                f"CloudWatch Log Group: {group_name}\n"
                f"Storage: {stored_bytes / 1024 / 1024:.2f} MB\n"
                f"Created: {created_dt.isoformat() if created_dt else 'unknown'}"
            )

            yield DataPoint(
                content=content,
                category=DataCategory.PROJECT,
                subcategory="log_group",
                source_type="aws_cloudwatch",
                source_id=group.get("arn", group_name),
                original_date=created_dt,
                topics=["aws", "cloudwatch", "logs", "monitoring"],
                raw_data=group,
            )

    async def _iterate_iam(
        self, loop: asyncio.AbstractEventLoop, account_id: str
    ) -> AsyncIterator[DataPoint]:
        """Iterate through IAM users and roles."""
        # Users
        users = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["iam", "list-users"])
        )

        if users and "Users" in users:
            for user in users["Users"]:
                user_name = user.get("UserName", "")
                create_date = user.get("CreateDate", "")
                arn = user.get("Arn", "")

                content = f"IAM User: {user_name}\nARN: {arn}\nCreated: {create_date}"

                yield DataPoint(
                    content=content,
                    category=DataCategory.PROJECT,
                    subcategory="iam_user",
                    source_type="aws_iam",
                    source_id=arn,
                    original_date=self._parse_aws_date(create_date),
                    topics=["aws", "iam", "security", "identity"],
                    raw_data=user,
                )

        # Roles
        roles = await loop.run_in_executor(
            None, lambda: self._run_aws_command(["iam", "list-roles"])
        )

        if roles and "Roles" in roles:
            for role in roles["Roles"]:
                role_name = role.get("RoleName", "")
                create_date = role.get("CreateDate", "")
                arn = role.get("Arn", "")
                description = role.get("Description", "")

                content = (
                    f"IAM Role: {role_name}\n"
                    f"Description: {description}\n"
                    f"ARN: {arn}\n"
                    f"Created: {create_date}"
                )

                yield DataPoint(
                    content=content,
                    category=DataCategory.PROJECT,
                    subcategory="iam_role",
                    source_type="aws_iam",
                    source_id=arn,
                    original_date=self._parse_aws_date(create_date),
                    topics=["aws", "iam", "security", "identity"],
                    raw_data=role,
                )

    def _parse_aws_date(self, date_str: str) -> datetime | None:
        """Parse AWS date formats."""
        if not date_str:
            return None

        try:
            # Try ISO format
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        try:
            # Try AWS format
            return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f%z")
        except ValueError:
            pass

        return None
