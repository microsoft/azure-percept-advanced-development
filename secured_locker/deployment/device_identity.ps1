# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


[CmdletBinding()]
param(
    [string]$subscription='',
    [string]$vaultName=''
)

function Connect-Azure {
  if (-not $silent) {
    Write-Host "`n Connecting to Azure account ..." -ForegroundColor Green
  }
  
  Connect-AzAccount > $null

  if ($subscription) {
      Set-AzContext -Subscription $subscription > $null
  }

  " Using subscription: {0}" -f (Get-AzContext).Subscription.name | Write-Host -ForegroundColor Cyan
}

function New-ServicePrincipal {
    
    Write-Host "`n Creating service principal ..." -ForegroundColor Green
    
    $sp = New-AzADServicePrincipal
    $global:sp_app_id = $sp.ApplicationId.ToString()
    $global:sp_obj_id = $sp.Id
    $global:sp_tenant = (Get-AzContext).Tenant.Id
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($sp.Secret)
    $global:sp_secret = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

    Set-AzKeyVaultAccessPolicy -VaultName $vaultName -ObjectId $sp.Id -PermissionsToSecrets get
}

function Write-Summary {
    Write-Host "`n Azure Percept model management server is provisioned at: ", $global:service_endpoint -ForegroundColor Cyan
    Write-Host " Service Principal Client ID:     ", $global:sp_app_id -ForegroundColor Cyan
    Write-Host " Service Principal Tenant ID:     ", $global:sp_tenant -ForegroundColor Cyan
    Write-Host " Service Principal Client Secret: ", $global:sp_secret -ForegroundColor Cyan
}

function New-DevicePrincipal {
    Connect-Azure
    New-ServicePrincipal
    Write-Summary
}

$WarningPreference = 'SilentlyContinue'
$global:sp_app_id=""
$global:sp_secret=""
$global:sp_tenant=""

New-DevicePrincipal
