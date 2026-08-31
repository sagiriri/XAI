$ErrorActionPreference = "Continue"

Write-Host "==============================================="
Write-Host " IRIS-XAI FULL OVERNIGHT BENCHMARK"
Write-Host "==============================================="

$datasets = @("cifar10", "funnybirds")
$models = @("simplecnn", "resnet18", "efficientnet")
$methods = @("gradcam", "shap", "lime")

foreach ($dataset in $datasets) {
    foreach ($model in $models) {
        foreach ($method in $methods) {

            $output = "benchmark_protocol_${dataset}_${model}_${method}.csv"
if (Test-Path ".\results\$output") {
    Write-Host "SKIPPING: $output already exists"
    continue
}

            Write-Host ""
            Write-Host "==============================================="
            Write-Host " Dataset : $dataset"
            Write-Host " Model   : $model"
            Write-Host " Method  : $method"
            Write-Host "==============================================="

            python .\src\benchmark_module.py `
                --dataset $dataset `
                --models $model `
                --num_xai_samples 60 `
                --xai_methods $method `
                --checkpoint_dir .\models `
                --output $output

            if ($LASTEXITCODE -ne 0) {
                Write-Host ""
                Write-Host "!!! FAILED: $dataset / $model / $method !!!"
                Write-Host "Continuing to the next configuration..."
            }
            else {
                Write-Host ""
                Write-Host "DONE: $dataset / $model / $method"
            }
        }
    }
}

Write-Host ""
Write-Host "==============================================="
Write-Host " OVERNIGHT BENCHMARK FINISHED"
Write-Host "==============================================="