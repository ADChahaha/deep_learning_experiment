Get-CimInstance Win32_Process |
    Where-Object { $_.Name -in @("powershell.exe", "python.exe") } |
    Select-Object ProcessId, ParentProcessId, Name, CommandLine |
    Format-List
