-- FUNCTION: public.internacao_by_periodo(character varying, character varying)

-- DROP FUNCTION public.internacao_by_periodo(character varying, character varying);

CREATE OR REPLACE FUNCTION public.internacao_by_periodo(
	dt_ini character varying,
	dt_fim character varying)
    RETURNS TABLE(person_id bigint, visit_id bigint, internacao_json text, internacao_str text) 
    LANGUAGE 'plpgsql'

    COST 100
    VOLATILE 
    ROWS 1000
AS $BODY$
DECLARE
  	v_person_id cdm5.person.person_id%type;
  	v_visit_id  cdm5.visit_occurrence.visit_occurrence_id%type;
  	v_registros_internacao_json   	text;
  	v_registros_internacao_str		text;	
  
  	rec_visit RECORD;
  
  	--Cursor that will get all the health care treatment in a period
	cur_visit_occurence 
  		CURSOR(c_dt_ini date, c_dt_fim date) FOR
			select  inter.person_id, inter.visit_occurrence_id, row_to_json(inter) as inter_json
					from (select vo.person_id,
								   vo.visit_occurrence_id,	   
								   vo.visit_concept_id,
								   (select con.concept_name
									  from cdm5.concept con
									 where con.concept_id = vo.visit_concept_id) as visit_concept_name,
								   vo.visit_start_date,
								   vo.visit_end_date,
								   vo.visit_source_value,
								   (select json_agg(ocorrencia)
								    from ( select co.condition_concept_id,
												 (select con.concept_name
									                from cdm5.concept con
									               where con.concept_id = co.condition_concept_id) as condition_ocurrence_concept_name,
								                 co.condition_start_date,
												 co.condition_end_date,
											     co.stop_reason,
											     co.visit_detail_id,
											     co.condition_source_concept_id,
											     co.condition_source_value
											from cdm5.condition_occurrence co
											where co.visit_occurrence_id = vo.visit_occurrence_id
										  ) as ocorrencia
									) ocorrencias,
						  		   (select json_agg(procedimento)
								      from ( select po.procedure_concept_id,
												    (select con.concept_name
													   from cdm5.concept con
													  where con.concept_id = po.procedure_concept_id) as procedure_ocurrence_concept_name,
											        po.procedure_date,
												    po.procedure_type_concept_id,
													po.quantity
											   from cdm5.procedure_occurrence po
											  where po.visit_occurrence_id = vo.visit_occurrence_id
											    and po.person_id = vo.person_id										  	
									  	   ) procedimento
								   ) as procedimentos
							  from cdm5.visit_occurrence vo							  
							 where vo.visit_start_date >= c_dt_ini
							   and vo.visit_end_date <= c_dt_fim) as inter;

BEGIN	
	OPEN cur_visit_occurence(dt_ini, dt_fim);
	LOOP
		--rec_visit := NULL;
		FETCH cur_visit_occurence INTO rec_visit;
		EXIT WHEN NOT FOUND;
		
		person_id := rec_visit.person_id;
		visit_id := rec_visit.visit_occurrence_id;	
		
		--internacao_json := v_registros_internacao_json;
		internacao_json := rec_visit.inter_json;
		internacao_str := null;
		
		RETURN NEXT;
	END LOOP;
	CLOSE cur_visit_occurence;

END; $BODY$;

ALTER FUNCTION public.internacao_by_periodo(character varying, character varying)
    OWNER TO omop;

