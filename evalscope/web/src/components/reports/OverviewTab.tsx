import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table from '@/components/ui/Table'
import Badge from '@/components/ui/Badge'
import { scoreColor } from '@/utils/colorScale'
import PlotlyChart from '@/components/charts/PlotlyChart'
import ReportSummaryStats from './ReportSummaryStats'
import JsonViewer from '@/components/common/JsonViewer'

interface Props {
  reports: ReportData[]
  reportName: string
  rootPath: string
  taskConfig?: Record<string, unknown>
  onDatasetClick?: (dataset: string) => void
}

export default function OverviewTab({ reports, reportName, rootPath, taskConfig, onDatasetClick }: Props) {
  const { t } = useLocale()
  const sequentialRows = useMemo(() => {
    return reports
      .map((r) => ({ dataset: r.dataset_name, metadata: r.metadata?.sequential_stop }))
      .filter((r) => Boolean(r.metadata))
  }, [reports])

  const tableData = useMemo(() => {
    return reports.map((r) => ({
      Dataset: r.dataset_name,
      Score: r.score,
      Samples: r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0,
    }))
  }, [reports])

  const columns = [
    {
      key: 'Dataset',
      label: 'Dataset',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const name = String(row.Dataset)
        if (onDatasetClick) {
          return (
            <button
              onClick={() => onDatasetClick(name)}
              className={
                'text-[var(--accent)] hover:underline cursor-pointer bg-transparent border-none ' +
                'p-0 font-inherit text-left'
              }
            >
              {name}
            </button>
          )
        }
        return name
      },
    },
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score)
        const norm = score > 1 ? score / 100 : score
        return (
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-[60px] min-w-[60px] rounded-full bg-[var(--border)] overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${Math.min(100, norm * 100)}%`,
                  background: scoreColor(norm),
                }}
              />
            </div>
            <span className="font-mono font-medium tabular-nums" style={{ color: scoreColor(norm) }}>
              {score.toFixed(4)}
            </span>
          </div>
        )
      },
    },
    {
      key: 'Samples',
      label: 'Samples',
      sortable: true,
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Samples).toLocaleString()}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* Summary Stats */}
      <ReportSummaryStats reports={reports} />

      {sequentialRows.length > 0 && (
        <Card title="Sequential Evaluation">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-left text-xs uppercase text-[var(--text-muted)]">
                  <th className="py-2 pr-4 font-medium">Dataset</th>
                  <th className="py-2 pr-4 font-medium">Verdict</th>
                  <th className="py-2 pr-4 font-medium">Method</th>
                  <th className="py-2 pr-4 font-medium">Samples</th>
                  <th className="py-2 pr-4 font-medium">Estimate</th>
                  <th className="py-2 pr-4 font-medium">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {sequentialRows.map(({ dataset, metadata }) => {
                  const tokenUsage = metadata?.token_usage?.consumed
                  const totalTokens = tokenUsage?.total_tokens ?? 0
                  const method = metadata?.method ?? metadata?.strategy ?? 'unknown'
                  const verdict = metadata?.verdict ?? metadata?.decision ?? 'unknown'
                  const ci = metadata?.ci_lower !== undefined && metadata?.ci_upper !== undefined
                    ? `[${formatPercent(metadata.ci_lower)}, ${formatPercent(metadata.ci_upper)}]`
                    : 'N/A'
                  const greyZoneLower = metadata?.grey_zone?.lower ?? metadata?.p_lo ?? metadata?.target_range?.[0]
                  const greyZoneUpper = metadata?.grey_zone?.upper ?? metadata?.p_hi ?? metadata?.target_range?.[1]
                  const greyZone = greyZoneLower !== undefined && greyZoneUpper !== undefined
                    ? `[${formatPercent(greyZoneLower)}, ${formatPercent(greyZoneUpper)}]`
                    : 'N/A'
                  const edgeHypotheses = metadata?.p0 !== undefined && metadata?.p1 !== undefined
                    ? `p0 ${formatPercent(metadata.p0)} | p1 ${formatPercent(metadata.p1)}`
                    : null
                  const roundedHypotheses = method === 'sprt'
                    && metadata?.sprt_h0_success_rate !== undefined
                    && metadata?.sprt_h1_success_rate !== undefined
                    ? `rounded ${formatPercent(metadata.sprt_h0_success_rate)} | ${
                      formatPercent(metadata.sprt_h1_success_rate)
                    }`
                    : null
                  return (
                    <tr key={dataset} className="border-b border-[var(--border)] last:border-b-0">
                      <td className="py-3 pr-4 font-medium">{dataset}</td>
                      <td className="py-3 pr-4">
                        <Badge variant={verdictVariant(verdict)}>{verdict}</Badge>
                      </td>
                      <td className="py-3 pr-4">
                        <div className="font-mono text-xs">{method}</div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          {metadata?.risk_assessment ?? 'risk N/A'} | target {formatPercent(metadata?.target)}
                        </div>
                      </td>
                      <td className="py-3 pr-4">
                        <div className="font-mono text-xs">
                          {formatNumber(metadata?.samples_scored)} / {formatNumber(metadata?.samples_total)}
                        </div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          budget {formatNumber(metadata?.sample_budget)}
                          {' | '}
                          skipped {formatNumber(metadata?.samples_skipped)}
                        </div>
                      </td>
                      <td className="py-3 pr-4">
                        <div className="font-mono text-xs">mean {formatPercent(metadata?.mean)}</div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          CI {ci} | grey zone {greyZone}
                        </div>
                        {(edgeHypotheses || roundedHypotheses) && (
                          <div className="mt-1 text-xs text-[var(--text-muted)]">
                            {edgeHypotheses}
                            {edgeHypotheses && roundedHypotheses ? ' | ' : ''}
                            {roundedHypotheses}
                          </div>
                        )}
                        {method === 'sprt' && metadata?.sprt_llr !== undefined && (
                          <div className="mt-1 font-mono text-xs text-[var(--text-muted)]">
                            llr {formatSignedNumber(metadata.sprt_llr)}
                          </div>
                        )}
                        {method === 'bayes' && (
                          <div className="mt-1 text-xs text-[var(--text-muted)]">
                            post below {formatPercent(metadata?.bayes_posterior_below)}
                            {' | '}
                            in {formatPercent(metadata?.bayes_posterior_within)}
                            {' | '}
                            above {formatPercent(metadata?.bayes_posterior_above)}
                          </div>
                        )}
                      </td>
                      <td className="py-3 pr-4">
                        <div className="font-mono text-xs">{formatNumber(totalTokens)}</div>
                        <div className="mt-1 text-xs text-[var(--text-muted)]">
                          in {formatNumber(tokenUsage?.input_tokens)}
                          {' | '}
                          out {formatNumber(tokenUsage?.output_tokens)}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Scores Table */}
      <Card title={t('single.datasetScoresTable')}>
        <Table
          columns={columns}
          data={tableData}
          defaultSort={{ key: 'Score', dir: 'desc' }}
        />
      </Card>

      {/* Radar Chart */}
      <PlotlyChart
        src={getChartUrl(rootPath, 'radar', { reportName })}
        height={400}
        title={t('single.radarChart')}
      />

      {/* Task Config */}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <Card title={t('reportDetail.taskConfig')} collapsible>
          <JsonViewer value={taskConfig} maxHeight={400} />
        </Card>
      )}
    </div>
  )
}

function formatPercent(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  return `${(value * 100).toFixed(1)}%`
}

function formatNumber(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  return value.toLocaleString()
}

function formatSignedNumber(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A'
  if (!Number.isFinite(value)) return value > 0 ? '+inf' : '-inf'
  return value >= 0 ? `+${value.toFixed(3)}` : value.toFixed(3)
}

function verdictVariant(verdict: string): 'default' | 'success' | 'warning' | 'danger' {
  const v = verdict.toLowerCase()
  if (v === 'pass' || v === 'go' || v === 'above') return 'success'
  if (v === 'fail' || v === 'no_go' || v === 'below') return 'danger'
  if (v === 'borderline' || v === 'within') return 'warning'
  return 'default'
}
