// Verbatim text of the norms behind the rule catalog (rules/*.json), keyed by rule_id.
// Every quote is checked word for word against adilet.zan.kz ([A] in case/norms/sources.md).
// A rule without an entry here is shown without a quote; never paraphrase a norm into this file.

const RULES_231 = {
  document: "Правила кассовых операций и инкассации, постановление Правления НБРК от 29.11.2019 № 231",
  short: "Постановление НБРК № 231",
};

export const NORMS = {
  KZ_NBK_231_P20_CLIENT_PRESENT: {
    ...RULES_231,
    quote: "Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.",
  },
  KZ_NBK_231_P22_LEAVE_DURING_SERVICE: {
    ...RULES_231,
    quote: "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.",
  },
};
