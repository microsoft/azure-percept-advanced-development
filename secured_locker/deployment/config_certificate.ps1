#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

param(
    [string]$certPassword='',
    [switch]$silent=$false,
    [string]$subscription='',
    [string]$certFile='',
    [string]$prefix='',
    [string]$resourceGroup='',
    [string]$location=''
)

function New-Certificate {
    if (-not $certPassword) {
        $global:cert_password = Read-Host -Prompt ' Enter a password for your service certificate private key '
    } else {
        $global:cert_password= $certPassword
    }
    if (-not $silent) {
        Write-Host "`n Creating a self-signed certificate for the service ..." -ForegroundColor Green
    }
    openssl req -x509 -sha256 -nodes -days 365 -newkey rsa:2048 -keyout privateKey.key -out appgwcert.crt -subj ("/C=US/ST=Washington/L=Redmond/O=Microsoft/OU=AED/CN=" + $prefix + "." + $location + ".cloudapp.azure.com" ) >$null 2>&1
    openssl pkcs12 -export -out appgwcert.pfx -inkey privateKey.key -in appgwcert.crt -password ("pass:" + $global:cert_password) >$null 2>&1
    
    $cert_path = Resolve-Path "./appgwcert.pfx"
    $global:cert_data = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($cert_path))
    
}

function Connect-Azure {
    if (-not $silent) {
      Write-Host "`n Connecting to Azure account ..." -ForegroundColor Green
    }
    
    Connect-AzAccount > $null
  
    if ($subscription) {
        Set-AzContext -Subscription $subscription > $null
    }
    if (-not $silent) {
      " Using subscription: {0}" -f (Get-AzContext).Subscription.name | Write-Host -ForegroundColor Cyan
    }
  }

  function New-Deployment {
    if (-not $silent) {
        Write-Host "`n Updating Azure Percept locker ..." -ForegroundColor Green
    }

    $today=Get-Date -Format "MM-dd-yyyy"
    $deploymentName="SCZDeployment"+"$today"

    $secureCertPassword = $global:cert_password
    
    $paramsObj = @{location = $location; `
        prefix = $prefix; `
        cert_data = $global:cert_data; `
    }
    $paramsObj.Add("cert_password", $secureCertPassword)
    $mergedParamsObj = Get-AzTemplateParameters "./certdeploy.parameters.json"  $paramsObj
    $outputs = New-AzResourceGroupDeployment -Name $deploymentName -ResourceGroupName $resourceGroup -TemplateFile .\certdeploy.json  -TemplateParameterObject $mergedParamsObj
    foreach ($key in $outputs.Outputs.keys) {
        if ($key -eq "serviceEndpoint") {
            $global:service_endpoint = $outputs.Outputs[$key].value
        }
    }
}

function Update-SantaCruzLocker {    
    if ($certFile) {
        $cert_path = Resolve-Path $certFile
        $global:cert_data = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($cert_path))
        if (-not $certPassword) {
          $global:cert_password = Read-Host -Prompt ' Enter a password for your service certificate private key '
        } else {
            $global:cert_password= $certPassword
        }
    } else {
        New-Certificate
    }
    Connect-Azure
    New-Deployment
    Write-Host "`n Azure Percept model management server is provisioned at: ", $global:service_endpoint -ForegroundColor Cyan     
}

function Get-AzTemplateParameters {
    param(
      [string]
      $ParametersFilePath,
  
      [hashtable]
      $TemplateParameterObject = @{}
    )
  
    if (!$ParametersFilePath) {
      return $TemplateParameterObject
    }
  
    $parameterFileJson = (Get-Content -Raw $ParametersFilePath | ConvertFrom-Json)
    $parameters = @{}
    $keys = ($parameterFileJson.parameters | get-member -MemberType NoteProperty | ForEach-Object {$_.Name})
    foreach ($key in $keys) {
      $parameters[$key] = $parameterFileJson.parameters.$key.value
    }
    foreach ($key in $TemplateParameterObject.Keys) {
      if ($parameters.ContainsKey($key)) {
        $parameters.Remove($key)
      }
    }
  
    return $parameters + $TemplateParameterObject
}

$WarningPreference = 'SilentlyContinue'
$global:cert_data=""
$global:cert_password=""

Update-SantaCruzLocker