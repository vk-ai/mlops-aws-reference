locals {
  container_port = 8080
  listener_port  = var.certificate_arn == null ? 80 : 443
  # Two identical services that differ only in image tag and the MLflow alias they serve.
  tracks = {
    stable = { image_tag = var.stable_image_tag, alias = "champion", desired_count = var.desired_count }
    canary = { image_tag = coalesce(var.canary_image_tag, var.stable_image_tag), alias = "canary", desired_count = var.canary_weight > 0 ? var.desired_count : 0 }
  }
}

resource "aws_ecs_cluster" "this" {
  name = var.project
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "scoring" {
  name              = "/ecs/${var.project}/scoring"
  retention_in_days = 30
}

resource "aws_security_group" "tasks" {
  name   = "${var.project}-tasks"
  vpc_id = var.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "tasks_from_alb" {
  security_group_id            = aws_security_group.tasks.id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = local.container_port
  to_port                      = local.container_port
}

resource "aws_vpc_security_group_egress_rule" "tasks_https" {
  security_group_id = aws_security_group.tasks.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
}

resource "aws_ecs_task_definition" "track" {
  for_each                 = local.tracks
  family                   = "${var.project}-scoring-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name         = "scoring"
    image        = "${aws_ecr_repository.scoring.repository_url}:${each.value.image_tag}"
    essential    = true
    portMappings = [{ containerPort = local.container_port, protocol = "tcp" }]
    environment = [
      { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
      { name = "SERVE_ALIAS", value = each.value.alias },
    ]
    readonlyRootFilesystem = false
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.scoring.name
        awslogs-region        = data.aws_region.current.region
        awslogs-stream-prefix = each.key
      }
    }
  }])
}

resource "aws_ecs_service" "track" {
  for_each               = local.tracks
  name                   = "scoring-${each.key}"
  cluster                = aws_ecs_cluster.this.id
  task_definition        = aws_ecs_task_definition.track[each.key].arn
  desired_count          = each.value.desired_count
  launch_type            = "FARGATE"
  enable_execute_command = false
  propagate_tags         = "SERVICE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.track[each.key].arn
    container_name   = "scoring"
    container_port   = local.container_port
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.scoring]
}
