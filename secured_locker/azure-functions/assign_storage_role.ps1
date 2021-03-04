#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

[CmdletBinding()]
param(
    [string]$subscription='',
    [string]$resourceGroup='',
    [string]$storageName='',
    [string]$clientId=''
)

function Get-ServicePrincipal {
        $sp = Get-AzADServicePrincipal -ApplicationId $clientId
        $global:sp_obj_id = $sp.Id
}

$global:sp_obj_id=""

Connect-AzAccount

Set-AzContext -Subscription $subscription

Get-ServicePrincipal

New-AzRoleAssignment -ObjectId $global:sp_obj_id -RoleDefinitionName "Storage Blob Data Reader" -ResourceGroupName $resourceGroup -ResourceName $storageName -ResourceType "Microsoft.Storage/storageAccounts"