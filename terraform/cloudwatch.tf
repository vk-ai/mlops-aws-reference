# Canary guardrails at the infrastructure level (the app-level ones live in CanaryRouter.analyse).
resource "aws_cloudwatch_metric_alarm" "canary_5xx" {
  alarm_name          = "${var.project}-canary-5xx"
  alarm_description   = "Canary target group returns 5xx: roll back by setting canary_weight = 0."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 3
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.scoring.arn_suffix
    TargetGroup  = aws_lb_target_group.track["canary"].arn_suffix
  }
  alarm_actions = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
}

resource "aws_cloudwatch_metric_alarm" "canary_latency" {
  alarm_name          = "${var.project}-canary-p95-latency"
  alarm_description   = "Canary p95 latency above 300 ms."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  threshold           = 0.3
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions = {
    LoadBalancer = aws_lb.scoring.arn_suffix
    TargetGroup  = aws_lb_target_group.track["canary"].arn_suffix
  }
  alarm_actions = var.alarm_topic_arn == null ? [] : [var.alarm_topic_arn]
}
