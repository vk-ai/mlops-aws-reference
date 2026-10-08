# Canary at the edge: one listener forwarding to two target groups by weight.
# `canary_weight` is the knob the release pipeline turns (10 → 25 → 50 → 100, or 0 to roll back),
# mirroring the in-process CanaryRouter in src/mlops_ref/canary.py.
resource "aws_security_group" "alb" {
  name   = "${var.project}-alb"
  vpc_id = var.vpc_id
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "tcp"
  from_port         = local.listener_port
  to_port           = local.listener_port
}

resource "aws_vpc_security_group_egress_rule" "alb_to_tasks" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.tasks.id
  ip_protocol                  = "tcp"
  from_port                    = local.container_port
  to_port                      = local.container_port
}

resource "aws_lb" "scoring" {
  name                       = "${var.project}-scoring"
  load_balancer_type         = "application"
  internal                   = false
  security_groups            = [aws_security_group.alb.id]
  subnets                    = var.public_subnet_ids
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "track" {
  for_each             = local.tracks
  name                 = "${var.project}-${each.key}"
  port                 = local.container_port
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = var.vpc_id
  deregistration_delay = 30
  health_check {
    path                = "/healthz"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "scoring" {
  load_balancer_arn = aws_lb.scoring.arn
  port              = local.listener_port
  protocol          = var.certificate_arn == null ? "HTTP" : "HTTPS"
  certificate_arn   = var.certificate_arn
  ssl_policy        = var.certificate_arn == null ? null : "ELBSecurityPolicy-TLS13-1-2-2021-06"

  default_action {
    type = "forward"
    forward {
      target_group {
        arn    = aws_lb_target_group.track["stable"].arn
        weight = 100 - var.canary_weight
      }
      target_group {
        arn    = aws_lb_target_group.track["canary"].arn
        weight = var.canary_weight
      }
      stickiness {
        enabled  = true
        duration = 3600
      }
    }
  }
}
