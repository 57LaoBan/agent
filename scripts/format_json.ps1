# Simple JSON pretty-printer that preserves non-ASCII characters.
# - re-indents with 2 spaces
# - decodes \uXXXX escapes back to real characters (CJK readability)
# - leaves other escapes (\n \" \\ ...) untouched, so embedded JSON-in-string stays valid
# Usage:  powershell -File scripts/format_json.ps1 path1.json path2.json ...

param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Paths)

function Decode-UnicodeEscapes {
    param([string]$Text)
    # only \uXXXX (not preceded by another backslash that would escape it)
    return [System.Text.RegularExpressions.Regex]::Replace(
        $Text,
        '(?<!\\)\\u([0-9a-fA-F]{4})',
        { param($m) [char][int]("0x" + $m.Groups[1].Value) }
    )
}

function Format-JsonText {
    param([string]$Json, [int]$IndentSize = 2)

    $sb = [System.Text.StringBuilder]::new()
    $indent = 0
    $inString = $false
    $escape = $false
    $unit = ' ' * $IndentSize

    for ($i = 0; $i -lt $Json.Length; $i++) {
        $c = $Json[$i]

        if ($escape) {
            [void]$sb.Append($c); $escape = $false; continue
        }
        if ($c -eq '\') {
            [void]$sb.Append($c); $escape = $true; continue
        }
        if ($c -eq '"') {
            $inString = -not $inString; [void]$sb.Append($c); continue
        }
        if ($inString) {
            [void]$sb.Append($c); continue
        }

        switch ($c) {
            '{' { [void]$sb.Append($c); $indent++; [void]$sb.Append("`r`n" + ($unit * $indent)) }
            '[' { [void]$sb.Append($c); $indent++; [void]$sb.Append("`r`n" + ($unit * $indent)) }
            '}' { $indent--; [void]$sb.Append("`r`n" + ($unit * $indent)); [void]$sb.Append($c) }
            ']' { $indent--; [void]$sb.Append("`r`n" + ($unit * $indent)); [void]$sb.Append($c) }
            ',' { [void]$sb.Append($c); [void]$sb.Append("`r`n" + ($unit * $indent)) }
            ':' { [void]$sb.Append($c); [void]$sb.Append(' ') }
            default {
                if ($c -ne ' ' -and $c -ne "`n" -and $c -ne "`r" -and $c -ne "`t") {
                    [void]$sb.Append($c)
                }
            }
        }
    }
    return $sb.ToString()
}

foreach ($p in $Paths) {
    if (-not (Test-Path $p)) {
        Write-Warning "skip: $p not found"
        continue
    }
    $raw = Get-Content -Raw -Encoding UTF8 -Path $p
    $decoded = Decode-UnicodeEscapes -Text $raw
    $pretty = Format-JsonText -Json $decoded
    # write back as UTF-8 without BOM
    [System.IO.File]::WriteAllText((Resolve-Path $p), $pretty, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "formatted: $p"
}
