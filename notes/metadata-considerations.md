# Metadata considerations
Things we should consider adding in the metadata of the documents to help in metadata filtering: { <br>
`text: str` <br>
`product_name: str` <br>
`filename: str` <br>
`date_added: datetime` <br>
`chemicals_present: List(dict('name', 'CAS number (if present)', 'UN number (if present)'))` <br>
`PFAS_status: boolean` <br>
}
